import time
import uuid
import threading
import queue
from datetime import datetime, timezone

import assemblyai as aai
from assemblyai.streaming.v3 import StreamingClient, StreamingClientOptions, StreamingEvents, StreamingParameters
import pyaudio

from speech_loop import AI_Assistant
from logger import JSONLogger

from setup.constants import (
    AAI_SAMPLE_RATE,
    ASR_MODEL,
    END_OF_TURN_THRESHOLD,
    MAX_TURN_SILENCE,
    MIN_TURN_SILENCE,
    TTS_COOLDOWN,
    TTS_MODEL,
    TTS_SPEED,
    VAD_THRESHOLD,
    VOICE_ID,
    VOICE_STABILITY,
    VOICE_SIMILARITY_BOOST,
)
import sounddevice as sd
import numpy as np


class AudioQueue:
    """Custom audio source that feeds PyAudio microphone data to AssemblyAI stream."""

    def __init__(self, sample_rate=16000, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_q = queue.Queue(maxsize=50)  # ~3 seconds of audio buffer at 16kHz
        self._stop_event = threading.Event()

    def __iter__(self):
        return self

    def __next__(self):
        # Check if we should stop
        if self._stop_event.is_set():
            raise StopIteration

        try:
            # Wait for audio chunks with a reasonable timeout
            chunk = self.audio_q.get(timeout=1.0)
            if chunk is None:  # Sentinel for stop
                raise StopIteration
            return chunk
        except queue.Empty:
            # since microphone is constantly reading. Raise to stop the stream.
            raise StopIteration

    def push_audio(self, data):
        """Push audio data from microphone."""
        try:
            self.audio_q.put_nowait(data)
        except queue.Full:
            # Drop oldest chunk if queue is full
            try:
                self.audio_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.audio_q.put_nowait(data)
            except queue.Full:
                pass  # Skip if still full

    def stop(self):
        """Signal the stream to stop."""
        try:
            self.audio_q.put_nowait(None)
        except queue.Full:
            pass
        self._stop_event.set()


class TerminalAssistant(AI_Assistant):
    def __init__(self):
        super().__init__()
        self.streaming_client = None

        # Initialize logging for terminal version
        self.logger = JSONLogger()
        self.conversation_id = str(uuid.uuid4())
        self.author_id = "terminal_user"

        # Interruption handling
        self.interrupt_tts = threading.Event()
        self.user_speech_detected = threading.Event()
        self.pending_bot_text = None

    def start_exam_and_listen(self):
        """Start the exam and begin listening"""
        super().start_exam_and_listen()

    def generate_audio(self, text, *, is_feedback: bool = False):
        print(f"\nAcabot: {text}")
        timestamp = datetime.now(timezone.utc)
        if is_feedback:
            self._schedule_coro(self.log_feedback_message(text, created_at=timestamp))
        else:
            self._schedule_coro(self.log_bot_message(text, created_at=timestamp))

        self.interrupt_tts.clear()
        self.user_speech_detected.clear()
        self.pending_bot_text = text

        self.speaking = True
        try:
            tts_stream = self.eleven.text_to_speech.stream(
                voice_id=VOICE_ID,
                output_format="pcm_16000",
                text=text,
                model_id=TTS_MODEL,
                voice_settings={
                    "stability": VOICE_STABILITY,
                    "similarity_boost": VOICE_SIMILARITY_BOOST,
                    "speed": TTS_SPEED
                }
            )

            interrupted = self._play_interruptible_pcm_stream(tts_stream, sample_rate=16000)

            if interrupted:
                time.sleep(0.8)
                if not self.user_speech_detected.is_set():
                    self.interrupt_tts.clear()
                    self.speaking = True
                    tts_stream = self.eleven.text_to_speech.stream(
                        voice_id=VOICE_ID,
                        output_format="pcm_16000",
                        text=self.pending_bot_text,
                        model_id=TTS_MODEL,
                        voice_settings={
                            "stability": VOICE_STABILITY,
                            "similarity_boost": VOICE_SIMILARITY_BOOST,
                            "speed": TTS_SPEED
                        }
                    )
                    self._play_interruptible_pcm_stream(tts_stream, sample_rate=16000)
        finally:
            self.speaking = False
            self.speaking_cooldown_until = time.time() + TTS_COOLDOWN
            self.pending_bot_text = None

    def _play_interruptible_pcm_stream(self, pcm_iter, sample_rate: int, channels: int = 1, dtype=np.int16, block_frames: int = 2048):
        interrupted = False
        try:
            with sd.RawOutputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype=dtype,
                blocksize=block_frames,
            ) as stream:
                for chunk in pcm_iter:
                    if self.interrupt_tts.is_set():
                        interrupted = True
                        break
                    if not chunk:
                        continue
                    stream.write(chunk)
        except Exception:
            pass

        return interrupted

    def start_transcription(self):
        if self.streaming_client:
            return

        self.streaming_client = StreamingClient(
            StreamingClientOptions(api_key=aai.settings.api_key, api_host="streaming.assemblyai.com")
        )
        self.streaming_client.on(StreamingEvents.Turn, self._on_turn)
        self.streaming_client.on(StreamingEvents.SpeechStarted, self._on_speech_started)

        self.streaming_client.connect(
            StreamingParameters(
                sample_rate=AAI_SAMPLE_RATE,
                end_of_turn_confidence_threshold=END_OF_TURN_THRESHOLD,
                speech_model=ASR_MODEL,
                format_turns=True,
                min_turn_silence=MIN_TURN_SILENCE,
                max_turn_silence=MAX_TURN_SILENCE,
                vad_threshold=VAD_THRESHOLD,
            )
        )

        self.audio_queue = AudioQueue(sample_rate=AAI_SAMPLE_RATE, chunk_size=1024)
        self.mic_thread = threading.Thread(target=self._read_microphone, daemon=True)
        self.stream_thread = threading.Thread(target=self._stream_audio, daemon=True)

        self.mic_thread.start()
        self.stream_thread.start()

    def _on_speech_started(self, _client, event):
        if self.speaking:
            self.interrupt_tts.set()

    def _should_accept_final(self, text: str) -> bool:
        """Override parent to allow interruptions during bot speech."""
        if self.exam_over or self.busy:
            return False

        now = time.time()

        # Allow interruption if bot is speaking
        if self.speaking:
            self.interrupt_tts.set()
            # Don't block the transcript - let it through to process the user's speech
        elif now < self.speaking_cooldown_until:
            return False

        normalized = " ".join(text.lower().split()).strip()
        if not normalized:
            return False

        # Check for exact match within recent 1.5s window
        if self.last_final_transcript == normalized and (now - self.last_final_time) < 1.5:
            return False

        # Update state
        self.last_final_transcript = normalized
        self.last_final_time = now
        return True

    def _read_microphone(self):
        """Read audio from microphone using PyAudio with proper buffer management."""
        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=AAI_SAMPLE_RATE,
                input=True,
                frames_per_buffer=1024
            )

            while not self.exam_over:
                try:
                    data = stream.read(1024, exception_on_overflow=False)
                    if data:
                        self.audio_queue.push_audio(data)
                except Exception as e:
                    if not self.exam_over:
                        print(f"Microphone read error: {e}")
                    break

            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception as e:
            print(f"Microphone initialization error: {e}")
        finally:
            self.audio_queue.stop()

    def _stream_audio(self):
        """Stream audio from queue to AssemblyAI."""
        try:
            self.streaming_client.stream(self.audio_queue)
        except Exception as e:
            if not self.exam_over:
                print(f"Streaming error: {e}")
        finally:
            self.stop_transcription()

    def stop_transcription(self):
        # Stop the audio queue first
        if hasattr(self, 'audio_queue'):
            try:
                self.audio_queue.stop()
            except Exception:
                pass

        # Then disconnect the streaming client
        if self.streaming_client:
            try:
                self.streaming_client.disconnect(terminate=True)
            except Exception:
                pass
            self.streaming_client = None

    # AssemblyAI turn handler
    def _on_turn(self, _client, event):
        if not getattr(event, "transcript", None):
            return

        if not event.end_of_turn:
            return

        transcript = event.transcript

        # Mark that user actually spoke (for interruption fallback)
        self.user_speech_detected.set()

        self.handle_final_transcript(transcript)
        self._schedule_coro(
            self.log_student_message(
                transcript,
                created_at=datetime.now(timezone.utc),
            )
        )

    def shutdown(self):
        # Log end of conversation
        self._run_coro(self.log_conversation_end())
        self.stop_transcription()
        super().shutdown()


if __name__ == "__main__":
    assistant = TerminalAssistant()
    try:
        assistant.start_exam_and_listen()
    finally:
        assistant.shutdown()