# path: ./real-time-translation.py
# title: 自動翻訳翻訳スクリプト

import os
# OpenMPの重複初期化エラーを回避
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# OpenMPとMKLのスレッド数を明示的に設定して競合を回避
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CT2_OMP_NUM_THREADS"] = "1"

import asyncio
import time
import argparse
import json
import sys
import multiprocessing
from collections import deque
from multiprocessing import shared_memory
import uuid

import numpy as np
import sounddevice as sd
import webrtcvad
import aiohttp
import psutil
import requests

# --- 基本設定 ---
TARGET_SAMPLE_RATE = 16000
WHISPER_MODEL_NAME = "small"
VAD_AGGRESSIVENESS = 2
VAD_FRAME_MS = 30
VAD_NUM_PADDING_FRAMES = 10
MIN_SPEECH_DURATION_S = 0.25
SILENCE_TIMEOUT_S = 0.5
MAX_SPEECH_DURATION_S = 10.0

# --- Ollama 設定 ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3n:e4b")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", OLLAMA_MODEL)
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
# モデルへの指示をより厳格にし、余計な情報（ローマ字、代替案など）の出力を抑制する
OLLAMA_SYSTEM_PROMPT = "Translate the following English text to Japanese. Output ONLY the translated Japanese text. Do not include romaji, explanations, or alternatives."
OLLAMA_CHAT_SYSTEM_PROMPT = "Translate the following text. If the input is Japanese, translate to English. If English, translate to Japanese. Output ONLY the translated text. Do not include romaji, explanations, or alternatives."
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
OLLAMA_WARMUP_TIMEOUT_S = 120

# --- 共有メモリ設定 ---
NUM_SHM_BLOCKS = 10
SHM_BLOCK_SIZE = int(MAX_SPEECH_DURATION_S * TARGET_SAMPLE_RATE * 4 * 1.1)

# -----------------------------------------------------------------------------
# ユーティリティ関数
# -----------------------------------------------------------------------------

def print_startup_info(args: argparse.Namespace):
    """起動時に設定情報を表示する"""
    print("==========================================================")
    print("      🎤 リアルタイム翻訳システム 🚀")
    print("==========================================================")
    print("\n[ 初期設定 ]")
    print("----------------------------------------------------------")
    print(f"  - Whisperモデル         : {args.model}")
    print(f"  - VADアグレッシブネス   : {args.vad_agg}")
    print(f"  - 無音タイムアウト      : {args.silence_timeout}秒")
    print(f"  - 最小発話時間        : {args.min_speech_duration}秒")
    print(f"  - 最大発話セグメント時間  : {MAX_SPEECH_DURATION_S}秒")
    print(f"  - Ollama URL            : {OLLAMA_BASE_URL}")
    print(f"  - 音声翻訳モデル        : {args.ollama_model}")
    print(f"  - チャット翻訳モデル      : {args.ollama_chat_model}")
    print(f"  - OMP_NUM_THREADS       : {os.environ.get('OMP_NUM_THREADS')}")
    print(f"  - MKL_NUM_THREADS       : {os.environ.get('MKL_NUM_THREADS')}")
    print(f"  - CT2_OMP_NUM_THREADS   : {os.environ.get('CT2_OMP_NUM_THREADS')}")
    print("----------------------------------------------------------\n")

def determine_optimal_threads() -> int:
    """使用可能なCPUコア数と空きメモリに基づいて、Whisper用の最適なスレッド数を決定する"""
    total_cores = os.cpu_count() or 1
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    if "OMP_NUM_THREADS" in os.environ:
        try:
            return int(os.environ["OMP_NUM_THREADS"])
        except ValueError:
            pass
    recommended_threads = max(1, total_cores // 2)
    if available_gb > 16: return min(recommended_threads, 8)
    if available_gb > 8: return min(recommended_threads, 4)
    if available_gb > 4: return min(recommended_threads, 2)
    return 1

# -----------------------------------------------------------------------------
# Whisper文字起こし専用プロセス
# -----------------------------------------------------------------------------

def run_transcriber_process(
    speech_queue: multiprocessing.Queue,
    text_queue: multiprocessing.Queue,
    free_shm_queue: multiprocessing.Queue,
    model_name: str
):
    """共有メモリ経由で音声データを受け取り、faster-whisperで文字起こしを行うプロセス"""
    try:
        import torch
        from faster_whisper import WhisperModel
        import numpy as np

        print("🧠 Whisperプロセス: モデルのロードを開始...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        num_threads = determine_optimal_threads()
        model = WhisperModel(model_name, device=device, compute_type=compute_type, cpu_threads=num_threads)
        print(f"✅ Whisperプロセス: モデルロード完了 (Device: {device.upper()}, Type: {compute_type}, Threads: {num_threads})")
        text_queue.put("WHISPER_READY")

    except Exception as e:
        error_message = f"FATAL:whisper_process:モデルの初期化に失敗しました: {type(e).__name__}: {e}"
        print(error_message, file=sys.stderr)
        text_queue.put(error_message)
        return

    while True:
        try:
            shm_info = speech_queue.get()
            if shm_info is None:
                print("🧠 Whisperプロセス: 終了シグナルを受信しました。")
                break

            shm_name, data_size, data_dtype_str = shm_info
            data_dtype = np.dtype(data_dtype_str)
            shm = None
            try:
                shm = shared_memory.SharedMemory(name=shm_name)
                audio_segment = np.copy(np.ndarray((data_size,), dtype=data_dtype, buffer=shm.buf))
                
                segments, _ = model.transcribe(
                    audio_segment,
                    language='en',
                    beam_size=5,
                    patience=1.0,
                    log_prob_threshold=-1.0,
                    no_speech_threshold=0.7,
                    compression_ratio_threshold=2.4,
                    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                )
                transcript = " ".join([seg.text for seg in segments]).strip()

                if transcript:
                    text_queue.put(f"[WHISPER] {transcript}")
                else:
                    text_queue.put("[WHISPER] (empty)")
            finally:
                if shm:
                    shm.close()
                free_shm_queue.put(shm_name)
        except Exception as e:
            text_queue.put(f"[FATAL] Whisperプロセスエラー: {type(e).__name__}: {e}")
            if 'shm_name' in locals():
                free_shm_queue.put(shm_name)
            break
    print("✅ Whisperプロセス: 正常に終了しました。")

# -----------------------------------------------------------------------------
# 音声処理クラス
# -----------------------------------------------------------------------------

class AudioRecorder:
    """マイクからの音声入力を管理するクラス"""
    def __init__(self, audio_chunk_queue: asyncio.Queue, samplerate: int, frame_duration_ms: int, loop: asyncio.AbstractEventLoop):
        self.audio_chunk_queue = audio_chunk_queue
        self.samplerate = samplerate
        self.frame_duration_ms = frame_duration_ms
        self.chunk_size = int(samplerate * frame_duration_ms / 1000)
        self.loop = loop
        self.stream = None
        print(f"🎙️ AudioRecorder: サンプルレート={self.samplerate}, フレーム期間={self.frame_duration_ms}ms, チャンクサイズ={self.chunk_size}")

    def _callback(self, indata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags):
        if status:
            self.loop.call_soon_threadsafe(self.audio_chunk_queue.put_nowait, (None, f"\n⚠️ オーディオコールバックで問題が発生: {status}"))
        if self.loop.is_running():
            try:
                self.audio_chunk_queue.put_nowait(('audio', indata[:, 0].copy()))
            except asyncio.QueueFull:
                pass

    def start(self):
        try:
            sd.check_input_settings()
            self.stream = sd.InputStream(samplerate=self.samplerate, blocksize=self.chunk_size, channels=1, dtype='float32', callback=self._callback)
            self.stream.start()
            print("✅ AudioRecorder: マイクストリームを開始しました。")
        except Exception as e:
            print(f"❌ 致命的エラー: マイクの起動に失敗しました。デバイスが接続されているか確認してください。: {e}", file=sys.stderr)
            raise

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            print("✅ AudioRecorder: マイクストリームを停止しました。")

# -----------------------------------------------------------------------------
# 非同期タスク
# -----------------------------------------------------------------------------

async def vad_processor(
    audio_chunk_queue: asyncio.Queue,
    speech_ipc_queue: multiprocessing.Queue,
    free_shm_queue: multiprocessing.Queue,
    shm_instances: dict,
    print_queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    args: argparse.Namespace
):
    """VADを行い、発話セグメントを共有メモリ経由で送信する"""
    vad = webrtcvad.Vad(args.vad_agg)
    silence_timeout_frames = int(args.silence_timeout * 1000 / VAD_FRAME_MS)
    max_speech_duration_frames = int(MAX_SPEECH_DURATION_S * 1000 / VAD_FRAME_MS)
    
    state = "IDLE"
    history_audio_chunks = deque(maxlen=VAD_NUM_PADDING_FRAMES)
    current_speech_chunks = []
    frames_of_silence_after_speech = 0
    
    await print_queue.put(f"🎯 VADプロセッサ: VADアグレッシブネス={args.vad_agg}, 無音タイムアウトフレーム={silence_timeout_frames}")
    
    last_audio_time = time.monotonic()

    async def send_segment_via_shm(speech_chunks, reason):
        segment_duration_s = (len(speech_chunks) * VAD_FRAME_MS) / 1000.0
        if segment_duration_s < args.min_speech_duration:
            await print_queue.put(f"🗑️ VAD: 短すぎる発話セグメントを破棄 (長さ: {segment_duration_s:.2f}秒)。")
            return

        shm_name = None
        try:
            shm_name = await loop.run_in_executor(None, free_shm_queue.get)
            speech_segment_np = np.concatenate(speech_chunks).astype(np.float32)
            
            shm = shm_instances[shm_name]
            
            if speech_segment_np.nbytes > shm.size:
                raise MemoryError(f"音声データ({speech_segment_np.nbytes} bytes)が共有メモリブロック({shm.size} bytes)を超えています。")
            
            shm.buf[:speech_segment_np.nbytes] = speech_segment_np.tobytes()
            
            shm_info = (shm_name, speech_segment_np.size, str(speech_segment_np.dtype))
            await loop.run_in_executor(None, speech_ipc_queue.put, shm_info)
            await print_queue.put(f"🚀 VAD: 発話セグメント ({segment_duration_s:.2f}秒, {reason}) を共有メモリ({shm_name})経由で送信。")
        except Exception as e:
            await print_queue.put(f"\n❌ VADプロセッサ: 共有メモリへの書き込み/送信エラー: {e}")
            if shm_name:
                await loop.run_in_executor(None, free_shm_queue.put, shm_name)

    while True:
        try:
            try:
                msg_type, data = await asyncio.wait_for(audio_chunk_queue.get(), timeout=5.0)
                if msg_type is None:
                    await print_queue.put(data)
                    continue

                audio_chunk_float32 = data
                last_audio_time = time.monotonic()
                max_vol = np.max(np.abs(audio_chunk_float32))
                if max_vol > 0.005:
                    await print_queue.put(('level', max_vol))
            except asyncio.TimeoutError:
                if time.monotonic() - last_audio_time > 5.0:
                    await print_queue.put("\n\n⚠️ VADプロセッサ: 5秒以上音声データを受信していません。マイクが正しく動作しているか、システム設定でマイクへのアクセスが許可されているか確認してください。\n")
                    last_audio_time = time.monotonic()
                continue

            audio_chunk_int16 = (audio_chunk_float32 * 32767).astype(np.int16)
            
            if audio_chunk_int16.nbytes != (TARGET_SAMPLE_RATE // 1000 * VAD_FRAME_MS * 2):
                history_audio_chunks.append(audio_chunk_float32)
                audio_chunk_queue.task_done()
                continue
            
            is_speech_now = vad.is_speech(audio_chunk_int16.tobytes(), TARGET_SAMPLE_RATE)
            
            if state == "IDLE":
                if is_speech_now:
                    await print_queue.put("\n")
                    state = "SPEAKING"
                    current_speech_chunks = list(history_audio_chunks) + [audio_chunk_float32]
                    await print_queue.put("🗣️ VAD: IDLE -> SPEAKING (発話開始)")
                else:
                    history_audio_chunks.append(audio_chunk_float32)
            elif state == "SPEAKING":
                current_speech_chunks.append(audio_chunk_float32)
                if not is_speech_now:
                    state = "PENDING_END"
                    frames_of_silence_after_speech = 1
                elif len(current_speech_chunks) >= max_speech_duration_frames:
                    await send_segment_via_shm(current_speech_chunks, "強制終了")
                    state = "IDLE"
                    current_speech_chunks.clear()
                    history_audio_chunks.clear()
                    await print_queue.put("🔄 VAD: SPEAKING -> IDLE (強制終了)")
            elif state == "PENDING_END":
                current_speech_chunks.append(audio_chunk_float32)
                if is_speech_now:
                    state = "SPEAKING"
                    frames_of_silence_after_speech = 0
                else:
                    frames_of_silence_after_speech += 1
                    if frames_of_silence_after_speech >= silence_timeout_frames:
                        speech_only_chunks = current_speech_chunks[:-frames_of_silence_after_speech]
                        await send_segment_via_shm(speech_only_chunks, "無音検出")
                        state = "IDLE"
                        current_speech_chunks.clear()
                        history_audio_chunks.clear()
                        await print_queue.put("🔄 VAD: PENDING_END -> IDLE (発話終了)")
            audio_chunk_queue.task_done()
        except asyncio.CancelledError:
            await print_queue.put("VADプロセッサ: キャンセルされました。")
            break
        except Exception as e:
            await print_queue.put(f"\n❌ VADプロセッサ: 予期せぬエラーが発生: {e}")

async def ollama_translator(text_queue: asyncio.Queue, print_queue: asyncio.Queue, session: aiohttp.ClientSession, args: argparse.Namespace):
    """文字起こしされたテキストをOllamaで翻訳する"""
    await print_queue.put("✅ Ollama音声翻訳タスクを開始しました。")
    while True:
        english_text = await text_queue.get()
        if english_text.startswith("[WHISPER]"):
            content = english_text.replace("[WHISPER]", "").strip()
            if content == "(empty)": continue
            
            await print_queue.put(f"🗣️ Transcription: {content}")
            payload = { "model": args.ollama_model, "stream": True, "messages": [{"role": "system", "content": args.ollama_prompt}, {"role": "user", "content": content}] }
            await _process_ollama_stream(session, f"{OLLAMA_BASE_URL}/api/chat", payload, "🌐 Translation: ", print_queue)
        text_queue.task_done()

async def chat_input_handler(chat_queue: asyncio.Queue, print_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    """ユーザーからのチャット入力を非同期で受け付ける"""
    while True:
        await print_queue.put(('prompt', None))
        message = await loop.run_in_executor(None, sys.stdin.readline)
        if message.strip():
            await chat_queue.put(message.strip())

async def ollama_chat_translator(chat_queue: asyncio.Queue, print_queue: asyncio.Queue, session: aiohttp.ClientSession, args: argparse.Namespace):
    """チャット入力をOllamaで翻訳する"""
    await print_queue.put("✅ Ollamaチャット翻訳タスクを開始しました。")
    while True:
        text_to_translate = await chat_queue.get()
        await print_queue.put(f"💬 You: {text_to_translate}")
        payload = { "model": args.ollama_chat_model, "stream": True, "messages": [{"role": "system", "content": args.ollama_chat_prompt}, {"role": "user", "content": text_to_translate}] }
        await _process_ollama_stream(session, f"{OLLAMA_BASE_URL}/api/chat", payload, "🤖 Chat Translation: ", print_queue)
        chat_queue.task_done()

async def _process_ollama_stream(session: aiohttp.ClientSession, url: str, payload: dict, output_prefix: str, print_queue: asyncio.Queue):
    """Ollama APIとのストリーミング通信を処理する共通関数"""
    full_translation = ""
    try:
        async with session.post(url, json=payload, headers={"Accept": "application/json"}) as resp:
            if resp.status == 200:
                async for chunk in resp.content:
                    if not chunk: continue
                    try:
                        for line in chunk.decode('utf-8').strip().split('\n'):
                            if not line: continue
                            data = json.loads(line)
                            content_part = data.get("response", "") if "response" in data else data.get("message", {}).get("content", "")
                            full_translation += content_part
                            await print_queue.put(('stream', f"{output_prefix}{full_translation}"))
                            if data.get("done"):
                                await print_queue.put("\n")
                                return
                    except (json.JSONDecodeError, UnicodeDecodeError): continue
            else:
                error_text = await resp.text()
                await print_queue.put(f"\n❌ Ollama APIエラー (ステータス {resp.status}): {error_text}")
    except Exception as e:
        await print_queue.put(f"\n❌ Ollama通信エラー: {e}")

async def ipc_text_queue_reader(ipc_queue: multiprocessing.Queue, asyncio_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    """プロセス間キューからテキストを読み取り、asyncioキューに渡すブリッジ"""
    print("Bridge: IPCテキストキューリーダーを開始しました。")
    while True:
        try:
            text = await loop.run_in_executor(None, ipc_queue.get)
            if text is None: break
            if isinstance(text, str) and text.startswith("[FATAL]"):
                print(f"❌ Whisperプロセスで致命的なエラー:\n   {text}", file=sys.stderr)
                loop.stop()
                break
            await asyncio_queue.put(text)
        except (asyncio.CancelledError, BrokenPipeError): break
        except Exception as e:
            print(f"❌ Bridge: IPCリーダーエラー: {e}", file=sys.stderr)
            break

async def console_printer(print_queue: asyncio.Queue):
    """コンソールへの出力を一元管理する"""
    while True:
        try:
            item = await print_queue.get()
            if isinstance(item, tuple):
                msg_type, content = item
                if msg_type == 'stream':
                    print(f"\r{content}", end="", flush=True)
                elif msg_type == 'level':
                    bar = '█' * int(content * 300)
                    print(f"\r🎤 Input level: [{bar:<50}] {content:.3f}", end="", flush=True)
                elif msg_type == 'prompt':
                    print("\n> ", end="", flush=True)
            else:
                print(f"\r{item}{' ' * 20}\n", end="", flush=True)

            print_queue.task_done()
        except asyncio.CancelledError:
            break

# -----------------------------------------------------------------------------
# メイン実行ループ
# -----------------------------------------------------------------------------

async def run_main_loop(args, speech_ipc_queue, text_ipc_queue, free_shm_queue, shm_instances):
    loop = asyncio.get_running_loop()

    audio_chunk_queue = asyncio.Queue(maxsize=300)
    transcribed_text_queue = asyncio.Queue(maxsize=20)
    chat_text_queue = asyncio.Queue(maxsize=10)
    print_queue = asyncio.Queue()

    audio_recorder = AudioRecorder(audio_chunk_queue, TARGET_SAMPLE_RATE, VAD_FRAME_MS, loop)
    try:
        audio_recorder.start()
    except Exception as e:
        await print_queue.put(f"❌ メインループ: オーディオレコーダーの起動に失敗: {e}")
        return

    await print_queue.put("==========================================================")
    await print_queue.put("                ✨ サービス起動 ✨")
    await print_queue.put("==========================================================")
    await print_queue.put("🎙️  マイクからの録音を開始しました。")
    await print_queue.put("⌨️  チャット翻訳が有効です。テキストを入力してEnterを押してください。")
    await print_queue.put("🛑 Ctrl+Cでプログラムを終了します。")
    await print_queue.put("----------------------------------------------------------\n")

    async with aiohttp.ClientSession() as session:
        all_tasks = {
            asyncio.create_task(console_printer(print_queue)),
            asyncio.create_task(vad_processor(audio_chunk_queue, speech_ipc_queue, free_shm_queue, shm_instances, print_queue, loop, args)),
            asyncio.create_task(ollama_translator(transcribed_text_queue, print_queue, session, args)),
            asyncio.create_task(chat_input_handler(chat_text_queue, print_queue, loop)),
            asyncio.create_task(ollama_chat_translator(chat_text_queue, print_queue, session, args)),
            asyncio.create_task(ipc_text_queue_reader(text_ipc_queue, transcribed_text_queue, loop))
        }
        try:
            await asyncio.gather(*all_tasks)
        except asyncio.CancelledError:
            print("\n✅ メインループ: キャンセルされました。")
        finally:
            print("\n🧹 クリーンアップ処理を開始します...")
            audio_recorder.stop()
            for task in all_tasks: task.cancel()
            await asyncio.gather(*all_tasks, return_exceptions=True)
            print("✅ メインプロセスの全タスクをクリーンアップしました。")

def warm_up_ollama(model_name: str, prompt: str, chat: bool):
    """Ollamaモデルを同期的にウォームアップする"""
    api_url = f"{OLLAMA_BASE_URL}/api/chat"
    print(f"🔥 Ollama{'チャット' if chat else '音声'}翻訳モデル ({model_name}) のウォームアップ中...")
    payload = { "model": model_name, "stream": False, "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": "Hello."}], "options": {"num_predict": 1} }
    try:
        response = requests.post(api_url, json=payload, timeout=OLLAMA_WARMUP_TIMEOUT_S)
        response.raise_for_status()
        print(f"✅ Ollama{'チャット' if chat else '音声'}翻訳モデル ({model_name}) の準備完了。")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Ollamaウォームアップ失敗 ({model_name}): {e}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="リアルタイム英語音声文字起こし＆日本語翻訳")
    parser.add_argument("--model", type=str, default=WHISPER_MODEL_NAME, help=f"Whisperモデル名 (デフォルト: {WHISPER_MODEL_NAME})")
    parser.add_argument("--vad_agg", type=int, default=VAD_AGGRESSIVENESS, choices=[0,1,2,3], help=f"VADのアグレッシブネス(0-3) (デフォルト: {VAD_AGGRESSIVENESS})")
    parser.add_argument("--silence_timeout", type=float, default=SILENCE_TIMEOUT_S, help=f"無音許容時間(秒) (デフォルト: {SILENCE_TIMEOUT_S})")
    parser.add_argument("--min_speech_duration", type=float, default=MIN_SPEECH_DURATION_S, help=f"最小発話時間(秒) (デフォルト: {MIN_SPEECH_DURATION_S})")
    parser.add_argument("--ollama_model", type=str, default=OLLAMA_MODEL, help=f"音声翻訳用Ollamaモデル (デフォルト: {OLLAMA_MODEL})")
    parser.add_argument("--ollama_prompt", type=str, default=OLLAMA_SYSTEM_PROMPT, help=f"音声翻訳用システムプロンプト")
    parser.add_argument("--ollama_chat_model", type=str, default=OLLAMA_CHAT_MODEL, help=f"チャット翻訳用Ollamaモデル (デフォルト: {OLLAMA_CHAT_MODEL})")
    parser.add_argument("--ollama_chat_prompt", type=str, default=OLLAMA_CHAT_SYSTEM_PROMPT, help=f"チャット翻訳用システムプロンプト")
    args = parser.parse_args()
    print_startup_info(args)

    shm_instances = {}
    transcriber_process = None
    try:
        for i in range(NUM_SHM_BLOCKS):
            try:
                shm = shared_memory.SharedMemory(create=True, size=SHM_BLOCK_SIZE)
                shm_instances[shm.name] = shm
            except FileExistsError:
                name = list(shm_instances.keys())[-1].name if shm_instances else f"rt_trans_{i}"
                shared_memory.SharedMemory(name=name).unlink()
                shm = shared_memory.SharedMemory(create=True, size=SHM_BLOCK_SIZE)
                shm_instances[shm.name] = shm
        print(f"✅ {len(shm_instances)}個の共有メモリブロック ({SHM_BLOCK_SIZE/1024/1024:.2f}MB/個) を作成しました。")
        
        speech_ipc_queue = multiprocessing.Queue()
        text_ipc_queue = multiprocessing.Queue()
        free_shm_queue = multiprocessing.Queue()
        for name in shm_instances.keys():
            free_shm_queue.put(name)

        transcriber_process = multiprocessing.Process(target=run_transcriber_process, args=(speech_ipc_queue, text_ipc_queue, free_shm_queue, args.model), daemon=True)
        transcriber_process.start()
        
        print("⏳ Whisperプロセスの準備を待機中...")
        try:
            message = text_ipc_queue.get(timeout=OLLAMA_WARMUP_TIMEOUT_S)
            if message != "WHISPER_READY":
                raise RuntimeError(f"Whisperプロセスからの予期せぬメッセージ: {message}")
            print("✅ Whisperプロセス準備完了。")
        except multiprocessing.queues.Empty:
            raise RuntimeError("Whisperプロセスの準備待機がタイムアウトしました。")

        if not warm_up_ollama(args.ollama_model, args.ollama_prompt, chat=False): sys.exit(1)
        if not warm_up_ollama(args.ollama_chat_model, args.ollama_chat_prompt, chat=True): sys.exit(1)

        asyncio.run(run_main_loop(args, speech_ipc_queue, text_ipc_queue, free_shm_queue, shm_instances))

    except KeyboardInterrupt:
        print("\n🛑 Ctrl+C を検知しました。シャットダウンします...")
    except Exception as e:
        print(f"\n❌ 致命的なエラーが発生しました: {e}", file=sys.stderr)
    finally:
        print("\n🧹 クリーンアップ処理を開始します...")
        if 'transcriber_process' in locals() and transcriber_process and transcriber_process.is_alive():
            if 'speech_ipc_queue' in locals():
                speech_ipc_queue.put(None)
            transcriber_process.join(timeout=5)
            if transcriber_process.is_alive():
                print("⚠️ Whisperプロセスが応答しないため、強制終了します。")
                transcriber_process.terminate()
                transcriber_process.join()
        
        for shm in shm_instances.values():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
        print(f"✅ 共有メモリブロックを解放しました。")
        print("\n✅ プログラムを完全に終了しました。")

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    main()