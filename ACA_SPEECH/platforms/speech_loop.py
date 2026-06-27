from abc import ABC, abstractmethod
import time
import threading
import asyncio
import os
from datetime import datetime, timezone

from dialog_instance import Dialog
from setup.constants import *
from setup.keys import ASSEMBLY_API_KEY, ELEVENLABS_API_KEY
from utils import merge_transcripts, normalize_text, is_near_duplicate
from logger import ExamLogMixin
import assemblyai as aai
from elevenlabs.client import ElevenLabs

if not ASSEMBLY_API_KEY:
    raise RuntimeError("AssemblyAI API key missing. Set setup.keys.ASSEMBLY_API_KEY.")

aai.settings.api_key = ASSEMBLY_API_KEY

class AI_Assistant(ABC, ExamLogMixin):
    def __init__(self):
        self._turn_lock = threading.Lock()

        # Keys and clients
        self.eleven = ElevenLabs(api_key=ELEVENLABS_API_KEY)

        # States
        self.busy = False
        self.exam_over = False
        self.speaking = False
        self.speaking_cooldown_until = 0.0

        # Dedupe
        self.last_final_transcript = ""
        self.last_final_time = 0.0

        # Latest measured model latency in seconds for the next bot message log
        self.last_model_latency_s = 0.0

        # Merge nearby final transcripts into one logical turn.
        self._pending_transcript = ""
        self._inflight_transcript = ""
        self._pending_timer = None
        self._pending_lock = threading.Lock()

        # Dialog engine
        self.dialog = Dialog(document, rubrics)

        # Async loop for Dialog
        self.loop = asyncio.new_event_loop()
        self._loop_ready = threading.Event()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()
        self._loop_ready.wait()

    # event loop
    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self._loop_ready.set()
        self.loop.run_forever()

    def _run_coro(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def _schedule_coro(self, coro):
        """Run a coroutine on the assistant's event loop without blocking."""
        try:
            asyncio.run_coroutine_threadsafe(coro, self.loop)
        except Exception:
            pass

    def _flush_pending_transcript(self):
        with self._pending_lock:
            transcript = self._pending_transcript
            self._pending_transcript = ""
            self._pending_timer = None

        if not transcript:
            return

        with self._turn_lock:
            if not self._should_accept_final(transcript):
                return
            self.busy = True

        threading.Thread(target=self._handle_turn, args=(transcript,), daemon=True).start()

    # exam flow
    def start_exam_and_listen(self):
        self.generate_audio("Hi, I am Acabot. We will start in a moment.")
        time.sleep(0.8)

        start = time.perf_counter()
        first_q = self._run_coro(self.dialog.send_message(""))
        self.last_model_latency_s = max(0.0, time.perf_counter() - start)
        if first_q:
            self.generate_audio(first_q)

        self.start_transcription()

        # Keep main thread alive while exam is running
        while not self.exam_over:
            time.sleep(0.1)

    # turn filtering
    def _should_accept_final(self, text: str) -> bool:
        if self.exam_over or self.busy:
            return False

        now = time.time()

        if self.speaking or now < self.speaking_cooldown_until:
            return False

        normalized = normalize_text(text)
        if not normalized:
            return False

        # Check for exact match within recent 1.5s window BEFORE updating state
        if self.last_final_transcript == normalized and (now - self.last_final_time) < 1.5:
            return False

        # Only update state if accepting (no duplicate found)
        self.last_final_transcript = normalized
        self.last_final_time = now
        return True

    def handle_final_transcript(self, transcript: str):
        """
        Subclasses call this when ASR produces a FINAL transcript.
        This function applies the shared busy/dedupe/cooldown rules and merges near-consecutive finals.
        """
        if not transcript:
            return

        with self._pending_lock:
            # Carry forward in-flight text if a turn is currently being processed
            if self.busy and self._inflight_transcript:
                self._pending_transcript = merge_transcripts(self._inflight_transcript, self._pending_transcript)

            normalized_new = normalize_text(transcript)
            self._pending_transcript = merge_transcripts(self._pending_transcript, transcript)

            if self._pending_timer is not None:
                self._pending_timer.cancel()

            merge_window_s = max(0.0, ASR_MERGE_WINDOW_MS / 1000.0)
            self._pending_timer = threading.Timer(merge_window_s, self._flush_pending_transcript)
            self._pending_timer.daemon = True
            self._pending_timer.start()

            if normalized_new and self.last_final_transcript and (time.time() - self.last_final_time) < 2.5:
                if is_near_duplicate(normalized_new, self.last_final_transcript, max_extra_words=3):
                    pending_started_at = None
                    pending_ended_at = None
                    pending_force_zero_latency = False
                    return

            candidates = self._pending_transcript.split(". ")
            for candidate in candidates:
                cand_norm = normalize_text(candidate)
                if is_near_duplicate(normalized_new, cand_norm, max_extra_words=3):
                    pending_started_at = None
                    pending_ended_at = None
                    pending_force_zero_latency = False
                    return

    def _handle_turn(self, text: str):
        self._inflight_transcript = text
        try:
            self.generate_ai_response(text)
        finally:
            self._inflight_transcript = ""
            self.busy = False
            self.speaking_cooldown_until = time.time() + 0.2

    def generate_ai_response(self, user_text: str):
        try:
            start = time.perf_counter()
            fut = asyncio.run_coroutine_threadsafe(self.dialog.send_message(user_text), self.loop)
            ai_response = fut.result(timeout=90) or ""
            self.last_model_latency_s = max(0.0, time.perf_counter() - start)
        except Exception:
            self.last_model_latency_s = 0.0
            ai_response = "Examiner: Sorry, something went wrong on my side. Could you repeat that?"

        is_feedback = "[examOver]" in ai_response
        if is_feedback:
            self.exam_over = True
            ai_response = ai_response.replace("[examOver]", "").strip()

        if ai_response:
            self.generate_audio(ai_response, is_feedback=is_feedback)

        if self.exam_over:
            try:
                self.stop_transcription()
            except Exception:
                pass
