# path: ./real-time-translation.py
# title: 自動翻訳翻訳スクリプト

import os
# OpenMPの重複初期化エラーを回避
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# OpenMPとMKLのスレッド数を明示的に設定して競合を回避
# クラッシュレポートでThread 13 (OpenMP) がクラッシュしているため、スレッド数を1に制限して試す
os.environ["OMP_NUM_THREADS"] = "1" #
os.environ["MKL_NUM_THREADS"] = "1" #
os.environ["CT2_OMP_NUM_THREADS"] = "1" # ctranslate2 specific

import asyncio
import time
import argparse
import json
import sys
import multiprocessing
from collections import deque

import numpy as np
import sounddevice as sd
import webrtcvad
import aiohttp
import psutil

# --- 基本設定 ---
TARGET_SAMPLE_RATE = 16000
WHISPER_MODEL_NAME = "base"
VAD_AGGRESSIVENESS = 2
VAD_FRAME_MS = 30
VAD_NUM_PADDING_FRAMES = 10
MIN_SPEECH_DURATION_S = 0.25
SILENCE_TIMEOUT_S = 0.5
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
# VADが検出する音声セグメントの最大長を制限する (IPC転送エラー対策)
# 長すぎるセグメントがNotImplementedErrorを引き起こしている可能性が高い
MAX_SPEECH_DURATION_S = 10.0 # 例えば10秒に制限
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

# --- Ollama 設定 ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3n:e4b")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", OLLAMA_MODEL)
OLLAMA_SYSTEM_PROMPT = "You are a helpful AI assistant. Please translate the following English text into short, natural Japanese. Only provide the Japanese translation."
OLLAMA_CHAT_SYSTEM_PROMPT = "You are a versatile AI assistant. Please translate the following text. If the input is Japanese, translate it to English. If the input is English, translate it to Japanese. Provide only the translated text."
OLLAMA_WARMUP_TIMEOUT_S = 30


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
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    print(f"  - 最大発話セグメント時間  : {MAX_SPEECH_DURATION_S}秒")
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    print(f"  - Ollama URL            : {OLLAMA_BASE_URL}")
    print(f"  - 音声翻訳モデル        : {args.ollama_model}")
    print(f"  - チャット翻訳モデル      : {args.ollama_chat_model}")
    print(f"  - OMP_NUM_THREADS       : {os.environ.get('OMP_NUM_THREADS')}")
    print(f"  - MKL_NUM_THREADS       : {os.environ.get('MKL_NUM_THREADS')}")
    print(f"  - CT2_OMP_NUM_THREADS   : {os.environ.get('CT2_OMP_NUM_THREADS')}")
    print("----------------------------------------------------------\n")

def determine_optimal_threads() -> int:
    """
    使用可能なCPUコア数と空きメモリに基づいて、Whisper用の最適なスレッド数を決定する。
    ただし、環境変数でOMP_NUM_THREADS等が設定されている場合、そちらが優先されるため、
    この関数は主に目安として使用される。
    """
    total_cores = os.cpu_count() or 1
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)

    # 環境変数でOpenMPスレッド数が設定されている場合はそちらを優先
    if "OMP_NUM_THREADS" in os.environ:
        try:
            return int(os.environ["OMP_NUM_THREADS"])
        except ValueError:
            pass

    recommended_threads = max(1, total_cores // 2) 

    if available_gb > 16:
        return min(recommended_threads, 8)
    if available_gb > 8:
        return min(recommended_threads, 4)
    if available_gb > 4:
        return min(recommended_threads, 2)
    return 1

# -----------------------------------------------------------------------------
# Whisper文字起こし専用プロセス
# -----------------------------------------------------------------------------

def run_transcriber_process(
    speech_queue: multiprocessing.Queue,
    text_queue: multiprocessing.Queue,
    model_name: str
):
    """
    faster-whisperモデルのロードと文字起こしを専門に行うプロセス。
    メモリ空間を分離することで、メインプロセスとのライブラリ競合を防ぐ。
    """
    try:
        import torch
        from faster_whisper import WhisperModel

        print("🧠 Whisperプロセス: モデルのロードを開始...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        num_threads = determine_optimal_threads()
        
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=num_threads
        )
        
        print(f"✅ Whisperプロセス: モデルロード完了 (Device: {device.upper()}, Type: {compute_type}, Threads: {num_threads})")

    except Exception as e:
        error_message = f"FATAL:whisper_process:モデルの初期化に失敗しました: {type(e).__name__}: {e}"
        print(error_message, file=sys.stderr)
        text_queue.put(error_message)
        return

    while True:
        try:
            audio_segment = speech_queue.get()
            if audio_segment is None:  # 終了シグナル
                print("🧠 Whisperプロセス: 終了シグナルを受信しました。")
                break
            
            segments, _ = model.transcribe(audio_segment, language='en', beam_size=5)
            transcript = " ".join([seg.text for seg in segments]).strip()
            
            if transcript:
                print(f"🧠 Whisperプロセス: 文字起こし結果: \"{transcript}\"")
                text_queue.put(transcript)
            else:
                print("🧠 Whisperプロセス: 文字起こし結果が空でした。")
        except Exception as e:
            print(f"⚠️ Whisperプロセスエラー: {type(e).__name__}: {e}", file=sys.stderr)
            error_message = f"FATAL:whisper_process:文字起こし中にエラーが発生しました: {type(e).__name__}: {e}"
            text_queue.put(error_message)
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
            print(f"⚠️ オーディオコールバックで問題が発生: {status}", file=sys.stderr)
        
        if self.loop.is_running():
            try:
                qsize = self.audio_chunk_queue.qsize()
                maxsize = self.audio_chunk_queue.maxsize
                if qsize > maxsize * 0.8 and qsize % 10 == 0:
                    print(f"⚠️ AudioQueue 混雑: {qsize}/{maxsize}", file=sys.stderr)
                elif qsize == maxsize and (qsize % 10 == 0 or qsize == maxsize):
                    print("⚠️ AudioQueue が満杯です。フレームをドロップしました。", file=sys.stderr)

                self.audio_chunk_queue.put_nowait(indata[:, 0].copy())
            except asyncio.QueueFull:
                pass

    def start(self):
        try:
            self.stream = sd.InputStream(
                samplerate=self.samplerate,
                blocksize=self.chunk_size, 
                channels=1,
                dtype='float32', 
                callback=self._callback
            )
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
    loop: asyncio.AbstractEventLoop,
    args: argparse.Namespace
):
    """VAD（音声区間検出）を行い、発話セグメントを切り出す"""
    vad = webrtcvad.Vad(args.vad_agg)
    silence_timeout_frames = int(args.silence_timeout * 1000 / VAD_FRAME_MS)
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    max_speech_duration_frames = int(MAX_SPEECH_DURATION_S * 1000 / VAD_FRAME_MS)
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    
    state = "IDLE"
    history_audio_chunks = deque(maxlen=VAD_NUM_PADDING_FRAMES) 
    current_speech_chunks = []
    frames_of_silence_after_speech = 0
    
    print(f"🎯 VADプロセッサ: VADアグレッシブネス={args.vad_agg}, 無音タイムアウトフレーム={silence_timeout_frames}")

    vad_process_times = deque(maxlen=100)
    vad_total_chunks_processed = 0

    while True:
        try:
            try:
                audio_chunk_float32 = await asyncio.wait_for(audio_chunk_queue.get(), timeout=VAD_FRAME_MS / 1000.0 * 2)
            except asyncio.TimeoutError:
                continue 
            
            vad_process_start_time = time.monotonic()

            audio_chunk_int16 = (audio_chunk_float32 * 32767).astype(np.int16)

            required_byte_size = (TARGET_SAMPLE_RATE // 1000 * VAD_FRAME_MS * 2)
            
            if audio_chunk_int16.nbytes != required_byte_size:
                history_audio_chunks.append(audio_chunk_float32) 
                audio_chunk_queue.task_done()
                continue

            audio_chunk_bytes = audio_chunk_int16.tobytes()

            try:
                is_speech_now = vad.is_speech(audio_chunk_bytes, TARGET_SAMPLE_RATE)
            except webrtcvad.Error as e:
                print(f"⚠️ VADエラー: {e}", file=sys.stderr)
                is_speech_now = False

            # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
            # MAX_SPEECH_DURATION_S を超えたら強制的に発話終了とみなすロジックを追加
            current_speech_duration_frames = len(current_speech_chunks)

            if state == "IDLE":
                if is_speech_now:
                    state = "SPEAKING"
                    current_speech_chunks = list(history_audio_chunks) + [audio_chunk_float32]
                    print("🗣️ VAD: IDLE -> SPEAKING (発話開始)")
                else:
                    history_audio_chunks.append(audio_chunk_float32)
            
            elif state == "SPEAKING":
                current_speech_chunks.append(audio_chunk_float32)
                if not is_speech_now:
                    state = "PENDING_END"
                    frames_of_silence_after_speech = 1
                
                # 最大発話時間を超えた場合、強制的にセグメントを終了
                elif current_speech_duration_frames >= max_speech_duration_frames:
                    print(f"⚠️ VAD: 最大発話時間 ({MAX_SPEECH_DURATION_S:.2f}秒) を超えました。強制的にセグメントを終了します。")
                    # 強制終了時は、現在の音声チャンクまでをセグメントとする
                    speech_only_chunks = current_speech_chunks
                    segment_duration_s = (len(speech_only_chunks) * VAD_FRAME_MS) / 1000.0
                    
                    if segment_duration_s >= args.min_speech_duration:
                        speech_segment_np = np.concatenate(speech_only_chunks)
                        print(f"🚀 VAD: 発話セグメントを検出 (長さ: {segment_duration_s:.2f}秒)。Whisperプロセスへ送信中... (強制終了)")
                        try:
                            await loop.run_in_executor(None, speech_ipc_queue.put, speech_segment_np)
                            print(f"🚀 VAD: 発話セグメントをWhisperキューにputしました。キューサイズ: {speech_ipc_queue.qsize()}")
                        except Exception as e:
                            print(f"❌ VADプロセッサ: IPCキューへのputでエラー: {type(e).__name__}: {e}", file=sys.stderr)
                            print("❌ VADプロセッサ: IPCキューが機能していません。システムを停止します。", file=sys.stderr)
                            loop.stop()
                            return
                    else:
                        print(f"🗑️ VAD: 短すぎる発話セグメントを破棄 (長さ: {segment_duration_s:.2f}秒, 最小: {args.min_speech_duration}秒)。")
                    
                    state = "IDLE"
                    current_speech_chunks.clear()
                    history_audio_chunks.clear()
                    print("🔄 VAD: SPEAKING -> IDLE (強制終了、状態リセット)")
            
            elif state == "PENDING_END":
                current_speech_chunks.append(audio_chunk_float32)
                if is_speech_now:
                    state = "SPEAKING"
                    frames_of_silence_after_speech = 0
                else:
                    frames_of_silence_after_speech += 1
                    if frames_of_silence_after_speech >= silence_timeout_frames:
                        speech_only_chunks = current_speech_chunks[:-frames_of_silence_after_speech]
                        
                        segment_duration_s = (len(speech_only_chunks) * VAD_FRAME_MS) / 1000.0
                        
                        if segment_duration_s >= args.min_speech_duration:
                            speech_segment_np = np.concatenate(speech_only_chunks)
                            print(f"🚀 VAD: 発話セグメントを検出 (長さ: {segment_duration_s:.2f}秒)。Whisperプロセスへ送信中...")
                            try:
                                await loop.run_in_executor(None, speech_ipc_queue.put, speech_segment_np)
                                print(f"🚀 VAD: 発話セグメントをWhisperキューにputしました。キューサイズ: {speech_ipc_queue.qsize()}")
                            except Exception as e:
                                print(f"❌ VADプロセッサ: IPCキューへのputでエラー: {type(e).__name__}: {e}", file=sys.stderr)
                                print("❌ VADプロセッサ: IPCキューが機能していません。システムを停止します。", file=sys.stderr)
                                loop.stop()
                                return
                        else:
                            print(f"🗑️ VAD: 短すぎる発話セグメントを破棄 (長さ: {segment_duration_s:.2f}秒, 最小: {args.min_speech_duration}秒)。")
                        
                        state = "IDLE"
                        current_speech_chunks.clear()
                        history_audio_chunks.clear()
                        print("🔄 VAD: PENDING_END -> IDLE (発話終了、状態リセット)")
            # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
            
            vad_process_end_time = time.monotonic()
            vad_process_time = vad_process_end_time - vad_process_start_time
            vad_process_times.append(vad_process_time)
            vad_total_chunks_processed += 1

            if vad_total_chunks_processed % 100 == 0:
                avg_vad_time = sum(vad_process_times) / len(vad_process_times) * 1000
                print(f"📊 VADプロセッサ: 過去100チャンクの平均処理時間: {avg_vad_time:.2f}ms (リアルタイム要求: {VAD_FRAME_MS:.2f}ms)")
                if avg_vad_time > VAD_FRAME_MS * 0.8:
                    print("⚠️ VADプロセッサ: 平均処理時間がリアルタイム要件に近づいています。ボトルネックの可能性があります。")

            audio_chunk_queue.task_done()

        except asyncio.CancelledError:
            print("VADプロセッサ: キャンセルされました。")
            break
        except Exception as e:
            print(f"❌ VADプロセッサ: 予期せぬエラーが発生: {e}", file=sys.stderr)
            audio_chunk_queue.task_done()
            break


async def _process_ollama_stream(session: aiohttp.ClientSession, url: str, payload: dict, output_prefix: str):
    """Ollama APIとのストリーミング通信を処理する共通関数"""
    full_translation = ""
    print(output_prefix, end="", flush=True)
    try:
        async with session.post(url, json=payload, headers={"Accept": "application/json"}) as resp:
            if resp.status == 200:
                async for chunk in resp.content:
                    if not chunk: continue
                    try:
                        for line in chunk.decode('utf-8').strip().split('\n'):
                            if not line: continue
                            data = json.loads(line)
                            content_part = ""
                            if "message" in data and "content" in data["message"]:
                                content_part = data["message"]["content"]
                            elif "response" in data: 
                                content_part = data["response"]
                            full_translation += content_part
                            print(f"\r{output_prefix}{full_translation}", end="", flush=True)
                            if "done" in data and data["done"]:
                                print()
                                return
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        continue
            else:
                error_text = await resp.text()
                print(f"\n❌ Ollama APIエラー (ステータス {resp.status}): {error_text}", file=sys.stderr)
                raise Exception(f"Ollama APIエラー: {error_text}")
    except aiohttp.ClientConnectorError as e:
        print(f"\n❌ Ollamaに接続できませんでした。Ollamaサーバが起動しているか、URL ({url}) が正しいか確認してください。エラー: {e}", file=sys.stderr)
    except asyncio.TimeoutError:
        print(f"\n❌ Ollamaからの応答がタイムアウトしました。サーバの負荷が高いか、モデルのロードに問題がある可能性があります。", file=sys.stderr)
    except Exception as e:
        print(f"\n❌ Ollama通信エラー: {e}", file=sys.stderr)

async def ollama_translator(text_queue: asyncio.Queue, session: aiohttp.ClientSession, args: argparse.Namespace):
    """文字起こしされたテキストをOllamaで翻訳する"""
    url = f"{OLLAMA_BASE_URL}/api/chat"
    
    print(f"🔥 Ollama音声翻訳モデル ({args.ollama_model}) のウォームアップ中...")
    warmup_payload = {
        "model": args.ollama_model,
        "stream": False,
        "messages": [{"role": "system", "content": args.ollama_prompt}, {"role": "user", "content": "Hello."}],
        "options": {"num_predict": 1}
    }
    try:
        timeout = aiohttp.ClientTimeout(total=OLLAMA_WARMUP_TIMEOUT_S)
        async with session.post(url, json=warmup_payload, timeout=timeout) as resp:
            if resp.status == 200:
                resp_json = await resp.json()
                if "message" in resp_json and "content" in resp_json["message"]:
                    pass 
                elif "response" in resp_json:
                    pass
                else:
                    print(f"⚠️ Ollamaウォームアップ: 予期しない応答形式。応答: {resp_json}", file=sys.stderr)
                print(f"✅ Ollama音声翻訳モデル ({args.ollama_model}) の準備完了。")
                print(f"翻訳を開始します。")
            else:
                error_text = await resp.text()
                print(f"⚠️ Ollamaウォームアップ失敗 (ステータス {resp.status}): {error_text}", file=sys.stderr)
                print("翻訳機能は動作しない可能性があります。", file=sys.stderr)
    except aiohttp.ClientConnectorError as e:
        print(f"❌ Ollamaウォームアップ中に接続エラー。Ollamaサーバが起動しているか確認してください。エラー: {e}", file=sys.stderr)
        print("翻訳機能は動作しない可能性があります。", file=sys.stderr)
    except asyncio.TimeoutError:
        print(f"❌ Ollamaウォームアップ中にタイムアウト ({OLLAMA_WARMUP_TIMEOUT_S}秒)。サーバの起動が遅いか、モデルのロードに時間がかかっています。", file=sys.stderr)
        print("翻訳機能は動作しない可能性があります。", file=sys.stderr)
    except Exception as e:
        print(f"❌ Ollamaウォームアップ中に予期せぬエラーが発生: {e}", file=sys.stderr)
        print("翻訳機能は動作しない可能性があります。", file=sys.stderr)

    while True:
        english_text = await text_queue.get()
        print(f"🗣️  Transcription: {english_text}")
        payload = {
            "model": args.ollama_model, "stream": True,
            "messages": [{"role": "system", "content": args.ollama_prompt}, {"role": "user", "content": english_text}]
        }
        await _process_ollama_stream(session, url, payload, "🌐 Translation: ")
        text_queue.task_done()

async def chat_input_handler(chat_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    """ユーザーからのチャット入力を非同期で受け付ける"""
    print("\n> ", end="", flush=True) 
    while True:
        message = await loop.run_in_executor(None, sys.stdin.readline)
        if message.strip():
            await chat_queue.put(message.strip())
        print("> ", end="", flush=True)

async def ollama_chat_translator(chat_queue: asyncio.Queue, session: aiohttp.ClientSession, args: argparse.Namespace):
    """チャット入力をOllamaで翻訳する"""
    url = f"{OLLAMA_BASE_URL}/api/chat"
    
    print(f"🔥 Ollamaチャット翻訳モデル ({args.ollama_chat_model}) のウォームアップ中...")
    chat_warmup_payload = {
        "model": args.ollama_chat_model,
        "stream": False,
        "messages": [{"role": "system", "content": args.ollama_chat_prompt}, {"role": "user", "content": "Hello."}],
        "options": {"num_predict": 1}
    }
    try:
        timeout = aiohttp.ClientTimeout(total=OLLAMA_WARMUP_TIMEOUT_S)
        async with session.post(url, json=chat_warmup_payload, timeout=timeout) as resp:
            if resp.status == 200:
                resp_json = await resp.json()
                if "message" in resp_json and "content" in resp_json["message"]:
                    pass 
                elif "response" in resp_json:
                    pass
                else:
                    print(f"⚠️ Ollamaチャットウォームアップ: 予期しない応答形式。応答: {resp_json}", file=sys.stderr)
                print(f"✅ Ollamaチャット翻訳モデル ({args.ollama_chat_model}) の準備完了。")
            else:
                error_text = await resp.text()
                print(f"⚠️ Ollamaチャットウォームアップ失敗 (ステータス {resp.status}): {error_text}", file=sys.stderr)
                print("チャット翻訳機能は動作しない可能性があります。", file=sys.stderr)
    except aiohttp.ClientConnectorError as e:
        print(f"❌ Ollamaチャットウォームアップ中に接続エラー。Ollamaサーバが起動しているか確認してください。エラー: {e}", file=sys.stderr)
        print("チャット翻訳機能は動作しない可能性があります。", file=sys.stderr)
    except asyncio.TimeoutError:
        print(f"❌ Ollamaチャットウォームアップ中にタイムアウト ({OLLAMA_WARMUP_TIMEOUT_S}秒)。", file=sys.stderr)
        print("チャット翻訳機能は動作しない可能性があります。", file=sys.stderr)
    except Exception as e:
        print(f"❌ Ollamaチャットウォームアップ中に予期せぬエラーが発生: {e}", file=sys.stderr)
        print("チャット翻訳機能は動作しない可能性があります。", file=sys.stderr)

    while True:
        text_to_translate = await chat_queue.get()
        print(f"💬 You: {text_to_translate}")
        payload = {
            "model": args.ollama_chat_model, "stream": True,
            "messages": [{"role": "system", "content": args.ollama_chat_prompt}, {"role": "user", "content": text_to_translate}]
        }
        await _process_ollama_stream(session, url, payload, "🤖 Chat Translation: ")
        chat_queue.task_done()

async def ipc_text_queue_reader(ipc_queue: multiprocessing.Queue, asyncio_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    """プロセス間キューからテキストを読み取り、asyncioキューに渡すブリッジ"""
    print("Bridge: IPCテキストキューリーダーを開始しました。")
    while True:
        try:
            try:
                text = await loop.run_in_executor(None, ipc_queue.get, 0.1)
            except multiprocessing.queues.Empty:
                await asyncio.sleep(0.01)
                continue
            
            if text is None: 
                print("Bridge: 終了シグナルを受信しました。")
                break
            if isinstance(text, str) and text.startswith("FATAL"):
                print(f"❌ Whisperプロセスで致命的なエラーが発生したため、メインプログラムを停止します:\n   {text}", file=sys.stderr)
                loop.stop()
                break
            print(f"Bridge: IPCキューからテキストを受信: \"{text}\" -> Asyncioキューへ転送")
            await asyncio_queue.put(text)
        except (asyncio.CancelledError, BrokenPipeError):
            print("Bridge: キャンセルまたはパイプ破損により終了します。")
            break
        except Exception as e:
            print(f"❌ Bridge: IPCリーダーエラー: {e}", file=sys.stderr)
            break

# -----------------------------------------------------------------------------
# メイン実行ループ
# -----------------------------------------------------------------------------

async def run_main_loop(args, speech_ipc_queue, text_ipc_queue):
    loop = asyncio.get_running_loop()

    audio_chunk_queue = asyncio.Queue(maxsize=300)
    transcribed_text_queue = asyncio.Queue(maxsize=20)
    chat_text_queue = asyncio.Queue(maxsize=10)

    audio_recorder = AudioRecorder(audio_chunk_queue, TARGET_SAMPLE_RATE, VAD_FRAME_MS, loop)
    try:
        audio_recorder.start()
    except Exception as e:
        print(f"❌ メインループ: オーディオレコーダーの起動に失敗しました。プログラムを終了します。: {e}", file=sys.stderr)
        return

    print("==========================================================")
    print("                ✨ サービス起動 ✨")
    print("==========================================================")
    print("🎙️  マイクからの録音を開始しました。")
    print("⌨️  チャット翻訳が有効です。テキストを入力してEnterを押してください。")
    print("🛑 Ctrl+Cでプログラムを終了します。")
    print("----------------------------------------------------------\n")

    async with aiohttp.ClientSession() as session:
        vad_task = asyncio.create_task(vad_processor(audio_chunk_queue, speech_ipc_queue, loop, args))
        translator_task = asyncio.create_task(ollama_translator(transcribed_text_queue, session, args))
        chat_input_task = asyncio.create_task(chat_input_handler(chat_text_queue, loop))
        chat_translator_task = asyncio.create_task(ollama_chat_translator(chat_text_queue, session, args))
        ipc_reader_task = asyncio.create_task(ipc_text_queue_reader(text_ipc_queue, transcribed_text_queue, loop))

        all_tasks = [vad_task, translator_task, chat_input_task, chat_translator_task, ipc_reader_task]
        
        try:
            await asyncio.gather(*all_tasks)
        except asyncio.CancelledError:
            print("\n✅ メインループ: キャンセルされました。")
        except Exception as e:
            print(f"\n❌ メインループ: 予期せぬエラーが発生: {e}", file=sys.stderr)
        finally:
            print("\n🧹 クリーンアップ処理を開始します...")
            audio_recorder.stop()
            for task in all_tasks:
                if task and not task.done():
                    task.cancel()
            await asyncio.gather(*all_tasks, return_exceptions=True)
            print("✅ メインプロセスの全タスクをクリーンアップしました。")

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

    speech_ipc_queue = multiprocessing.Queue()
    text_ipc_queue = multiprocessing.Queue()

    transcriber_process = multiprocessing.Process(
        target=run_transcriber_process,
        args=(speech_ipc_queue, text_ipc_queue, args.model),
        daemon=True
    )
    transcriber_process.start()
    
    print("⏳ Whisperプロセスの起動とモデルロードを待機中...")
    start_time = time.time()
    
    while transcriber_process.is_alive() and time.time() - start_time < OLLAMA_WARMUP_TIMEOUT_S:
        if not text_ipc_queue.empty():
            try:
                message = text_ipc_queue.get_nowait()
                if isinstance(message, str) and message.startswith("FATAL:whisper_process:"):
                    print(f"❌ Whisperプロセスの起動に失敗しました。プログラムを終了します。\n   詳細: {message}", file=sys.stderr)
                    transcriber_process.join(timeout=1)
                    return
            except multiprocessing.queues.Empty:
                pass 
        time.sleep(0.5)

    if not transcriber_process.is_alive():
        print("❌ Whisperプロセスが予期せず終了しました。IPCキューへのデータ投入に失敗する可能性があります。プログラムを終了します。", file=sys.stderr)
        if text_ipc_queue.empty():
            print("   詳細: Whisperプロセスからのエラーメッセージはありませんでした。原因不明。", file=sys.stderr)
        else:
            while not text_ipc_queue.empty():
                try:
                    error_msg = text_ipc_queue.get_nowait()
                    print(f"   詳細: {error_msg}", file=sys.stderr)
                except multiprocessing.queues.Empty:
                    break
        return

    try:
        asyncio.run(run_main_loop(args, speech_ipc_queue, text_ipc_queue))
    except KeyboardInterrupt:
        print("\n🛑 Ctrl+C を検知しました。シャットダウンします...")
    except Exception as e:
        print(f"❌ 予期せぬエラーが発生しました: {e}", file=sys.stderr)
    finally:
        if 'speech_ipc_queue' in locals():
            print("メイン: Whisperプロセスに終了シグナルを送っています...")
            try:
                speech_ipc_queue.put(None)
            except Exception as e:
                print(f"⚠️ メイン: speech_ipc_queue.put(None) でエラー: {e}", file=sys.stderr)
        
        if 'transcriber_process' in locals() and transcriber_process.is_alive():
             print("🧠 Whisperプロセスの終了を待っています...")
             transcriber_process.join(timeout=5)
             if transcriber_process.is_alive():
                print("⚠️ Whisperプロセスが応答しないため、強制終了します。")
                transcriber_process.terminate()
                transcriber_process.join(timeout=2)
        
        print("\n✅ プログラムを完全に終了しました。")

if __name__ == "__main__":
    # multiprocessingの開始メソッドをコメントアウトして、システムのデフォルトを使用
    # macOSで"spawn"が推奨されるものの、特定の環境やライブラリとの相性問題がある場合があるため、検証目的で試す
    # try:
    #     multiprocessing.set_start_method("spawn")
    # except RuntimeError:
    #     pass
    main()