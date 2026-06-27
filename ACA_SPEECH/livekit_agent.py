import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from livekit import agents, api, rtc
from livekit.agents import Agent, AgentSession, TurnHandlingOptions, JobProcess
from livekit.plugins import assemblyai, elevenlabs, silero

from logger import JSONLogger, ExamLogMixin
from setup.constants import (
    ASR_MODEL,
    AAI_SAMPLE_RATE,
    ASR_MERGE_WINDOW_MS,
    MAX_TURN_SILENCE,
    MIN_TURN_SILENCE,
    VAD_THRESHOLD,
    VOICE_ID,
    VOICE_STABILITY,
    VOICE_SIMILARITY_BOOST,
    TTS_MODEL,
    TTS_SPEED,
)
from setup.keys import LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, ASSEMBLY_API_KEY
from utils import merge_transcripts, normalize_text, is_near_duplicate

load_dotenv()

AGENT_NAME = os.getenv("AGENT_NAME", "").strip()

_WARMED = False
_WARMUP_LOCK = asyncio.Lock()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load(
        activation_threshold=VAD_THRESHOLD
    )

    try:
        from dialog_instance import get_global_vector_db

        print("[PREWARM] Loading global vector DB...")
        proc.userdata["vector_db"] = get_global_vector_db()
        print("[PREWARM] Global vector DB ready.")

    except Exception as e:
        print(f"[PREWARM] VectorDB warmup failed: {e}")

# Dialog is imported lazily in ExamSession to keep process startup light.
def _extract_job_metadata(ctx: agents.JobContext) -> dict:
    """Best-effort extraction of JSON metadata attached to a dispatch job."""
    job = getattr(ctx, "job", None)
    raw = getattr(job, "metadata", None)

    if not raw:
        return {}

    if isinstance(raw, dict):
        return raw

    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    return {}


async def _dispatch_create(room_name: str, metadata: str = "") -> None:
    if not AGENT_NAME:
        raise RuntimeError(
            "LIVEKIT_AGENT_NAME is not set. Set it before using dispatch-create."
        )

    lkapi = api.LiveKitAPI()
    try:
        dispatch = await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=AGENT_NAME,
                room=room_name,
                metadata=metadata or "{}",
            )
        )
        print("created dispatch", dispatch)

        dispatches = await lkapi.agent_dispatch.list_dispatch(room_name=room_name)
        print(f"there are {len(dispatches)} dispatches in {room_name}")
    finally:
        await lkapi.aclose()


async def _dispatch_list(room_name: str) -> None:
    lkapi = api.LiveKitAPI()
    try:
        dispatches = await lkapi.agent_dispatch.list_dispatch(room_name=room_name)
        print(f"there are {len(dispatches)} dispatches in {room_name}")
        for d in dispatches:
            print(d)
    finally:
        await lkapi.aclose()


def _build_voice_settings():
    """Build plugin-compatible ElevenLabs voice settings."""
    try:
        return elevenlabs.VoiceSettings(
            stability=VOICE_STABILITY,
            similarity_boost=VOICE_SIMILARITY_BOOST,
            speed=TTS_SPEED,
        )
    except TypeError:
        # Some plugin versions do not expose speed on VoiceSettings.
        return elevenlabs.VoiceSettings(
            stability=VOICE_STABILITY,
            similarity_boost=VOICE_SIMILARITY_BOOST,
        )


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are Acabot, a spoken exam assistant. "
                "The application controls the exam flow."
            )
        )


class ExamSession(ExamLogMixin):
    def __init__(self, author_id: str = "playground_user", vector_db=None):
        self.log_mode = "livekit"
        self.dialog = None
        self._dialog_init_lock = asyncio.Lock()
        self.logger = JSONLogger()
        self.conversation_id = str(uuid.uuid4())
        self.author_id = author_id
        self.audio_student_id = author_id
        self.exam_over = False
        self.busy = False
        self.last_model_latency_s = 0.0

        self.vector_db = vector_db

        # Needed by _should_accept_final()
        self.speaking = False
        self.speaking_cooldown_until = 0.0
        self.last_final_transcript = ""
        self.last_final_time = 0.0

        self._turn_lock = asyncio.Lock()

    def _should_accept_final(self, text: str) -> bool:
        if self.exam_over or self.busy:
            return False

        now = time.time()

        if getattr(self, "speaking", False) or now < getattr(self, "speaking_cooldown_until", 0.0):
            return False

        normalized = normalize_text(text)
        if not normalized:
            return False

        if (
            self.last_final_transcript == normalized
            and (now - self.last_final_time) < 1.5
        ):
            return False

        self.last_final_transcript = normalized
        self.last_final_time = now
        return True

    async def _ensure_dialog(self):
        if self.dialog is not None:
            return self.dialog

        async with self._dialog_init_lock:
            if self.dialog is None:
                # Import and initialize in a worker thread to avoid blocking the event loop.
                def _create_dialog():
                    from setup.constants import document, rubrics
                    from dialog_instance import Dialog
                    return Dialog(document, rubrics, vector_db=self.vector_db)

                self.dialog = await asyncio.to_thread(_create_dialog)

        return self.dialog

    async def prewarm(self):
        """Best-effort background warmup to reduce first-turn latency."""
        try:
            await self._ensure_dialog()
        except Exception:
            pass

    async def start_exam(self) -> str:
        dialog = await self._ensure_dialog()
        start = time.perf_counter()
        first_q = (await dialog.send_message("")) or ""
        self.last_model_latency_s = max(0.0, time.perf_counter() - start)
        return first_q

    async def handle_user_turn(self, transcript: str) -> str:
        async with self._turn_lock:
            if not self._should_accept_final(transcript):
                return ""

            self.busy = True
            snapshot = None

            try:
                dialog = await self._ensure_dialog()
                snapshot = dialog.snapshot_state()

                start = time.perf_counter()

                ai_response = await asyncio.wait_for(
                    dialog.send_message(transcript),
                    timeout=90,
                )

                ai_response = ai_response or ""
                self.last_model_latency_s = max(0.0, time.perf_counter() - start)

            except asyncio.CancelledError:
                if snapshot is not None:
                    dialog.restore_state(snapshot)
                raise


            except Exception as e:

                import traceback

                print("[HANDLE_USER_TURN ERROR]", repr(e))

                traceback.print_exc()

                if snapshot is not None:

                    try:

                        dialog.restore_state(snapshot)

                    except Exception as restore_error:

                        print("[RESTORE ERROR]", repr(restore_error))

                        traceback.print_exc()

                self.last_model_latency_s = 0.0

                ai_response = (

                    "Sorry, something went wrong on my side. "

                    "Could you repeat that?"

                )

            finally:
                self.busy = False

            if "[examOver]" in ai_response:
                self.exam_over = True
                ai_response = ai_response.replace("[examOver]", "").strip()

            return ai_response

    async def rollback_blocked_reply(self, reply_text: str) -> None:
         if not reply_text:
             return
         dialog = await self._ensure_dialog()
         undo_fn = getattr(dialog, "undo_last_examiner_message", None)
         if callable(undo_fn):
             try:
                 undo_fn(reply_text)
             except Exception:
                 pass

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    metadata = _extract_job_metadata(ctx)
    author_id = str(
        metadata.get("user_id") or metadata.get("username") or "playground_user"
    )

    # Fallback to actual participant identity in logs.
    if author_id == "playground_user":
        try:
            if ctx.room.remote_participants:
                first_participant = next(iter(ctx.room.remote_participants.values()))
                author_id = first_participant.identity or author_id
        except Exception:
            pass

    exam = ExamSession(
        author_id=author_id,
        vector_db=ctx.proc.userdata.get("vector_db"),
    )

    audio_recording_tasks = []

    async def _record_student_audio(track, participant_identity: str):
        sink = exam.logger.get_student_audio_sink(
            conversation_id=exam.conversation_id,
            student_id=participant_identity,
        )

        stream = rtc.AudioStream.from_track(
            track=track,
            sample_rate=16000,
            num_channels=1,
        )

        try:
            async for frame_event in stream:
                sink.write_frame(frame_event.frame)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[STUDENT AUDIO LOG ERROR] {participant_identity}: {e}")

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return

        participant_identity = getattr(participant, "identity", None) or exam.author_id
        exam.audio_student_id = participant_identity

        task = asyncio.create_task(
            _record_student_audio(track, participant_identity)
        )
        audio_recording_tasks.append(task)

    if not ASSEMBLY_API_KEY:
        raise RuntimeError("AssemblyAI API key missing. Set setup.keys.ASSEMBLY_API_KEY.")

    stt_engine = assemblyai.STT(
        api_key=ASSEMBLY_API_KEY,
        model=ASR_MODEL,
        sample_rate=AAI_SAMPLE_RATE,
        min_turn_silence=MIN_TURN_SILENCE,
        max_turn_silence=MAX_TURN_SILENCE,
        vad_threshold=VAD_THRESHOLD,
    )

    session = AgentSession(
        stt=stt_engine,
        tts=elevenlabs.TTS(
            voice_id=VOICE_ID,
            model=TTS_MODEL,
            language="en",
            voice_settings=_build_voice_settings(),
        ),
        vad=ctx.proc.userdata.get("vad") or silero.VAD.load(
            activation_threshold=VAD_THRESHOLD
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection="stt",
            min_endpointing_delay=0,
            max_endpointing_delay=0,
            min_interruption_duration=0.8,
            min_interruption_words=2,
        ),
        use_tts_aligned_transcript=True,
    )

    processing_task = None
    merge_task = None
    pending_transcript = ""
    pending_started_at = None
    pending_ended_at = None
    pending_force_zero_latency = False
    active_transcript = ""
    active_version = 0
    inflight_transcript = ""
    inflight_version = 0
    merge_window_s = max(0.0, ASR_MERGE_WINDOW_MS / 1000.0)
    request_version = 0
    session_running = False
    accept_user_input = False
    bot_speaking = False
    last_bot_speech_end_at = None
    last_processed_transcript = ""
    last_processed_time = 0.0

    def _ts(dt_obj: Optional[datetime]) -> str:
        return dt_obj.isoformat() if dt_obj is not None else "None"

    async def _safe_session_say(text: str, allow_interruptions: bool, add_to_chat_ctx: bool = True) -> bool:
        nonlocal session_running
        if not session_running:
            return False
        try:
            await session.say(
                text,
                allow_interruptions=allow_interruptions,
                add_to_chat_ctx=add_to_chat_ctx,
            )
            return True
        except RuntimeError as e:
            msg = str(e)
            if "isn't running" in msg or "is closing" in msg:
                session_running = False
                return False
            raise
        except asyncio.CancelledError:
            return False

    async def _cancel_processing_task():
        nonlocal processing_task
        if processing_task is None or processing_task.done():
            return
        processing_task.cancel()
        try:
            await processing_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        finally:
            processing_task = None

    async def _process_transcript(
            version,
            transcript,
            started_at: Optional[datetime],
            ended_at: Optional[datetime],
            force_zero_latency: bool,
    ):
        nonlocal request_version, active_transcript, active_version
        nonlocal last_processed_transcript, last_processed_time
        nonlocal inflight_transcript, inflight_version
        nonlocal bot_speaking, last_bot_speech_end_at
        nonlocal accept_user_input, pending_transcript, merge_task, session_running

        if exam.exam_over or version != request_version or not session_running:
            return

        inflight_transcript = transcript
        inflight_version = version

        normalized = normalize_text(transcript)
        now = time.time()

        if normalized and normalized == last_processed_transcript and (now - last_processed_time) < 1.5:
            return

        active_transcript = transcript
        active_version = version

        try:
            student_latency_override = None

            if force_zero_latency:
                student_latency_override = 0.0
            elif started_at is not None and last_bot_speech_end_at is not None:
                delta = (started_at - last_bot_speech_end_at).total_seconds()
                student_latency_override = max(0.0, delta)

            reply = await exam.handle_user_turn(transcript)

            print(f"[END DEBUG] exam_over={exam.exam_over}, reply={reply!r}")

            # IMPORTANT: close input gate immediately when exam ends
            if exam.exam_over:
                accept_user_input = False
                pending_transcript = ""

                if merge_task is not None and not merge_task.done():
                    merge_task.cancel()


        except asyncio.CancelledError:
            return

        finally:
            if active_version == version:
                active_transcript = ""

            if inflight_version == version:
                inflight_transcript = ""

        if normalized:
            last_processed_transcript = normalized
            last_processed_time = time.time()

        if version != request_version or not session_running:
            return

        if not reply:
            return

        bot_started_at = datetime.now(timezone.utc)

        # Final exam branch: speak feedback and then close the session.
        if exam.exam_over:
            accept_user_input = False
            pending_transcript = ""
            request_version += 1

            if merge_task is not None and not merge_task.done():
                merge_task.cancel()

            # STOP RECORDING HERE, before feedback is logged/spoken
            exam.logger.close_student_audio_sinks(
                exam.conversation_id,
                exam.audio_student_id,
            )

            await exam.log_student_message(
                transcript,
                created_at=started_at,
                response_time_override=0.0 if force_zero_latency else student_latency_override,
            )

            await exam.log_feedback_message(
                reply,
                created_at=datetime.now(timezone.utc),
            )

            await exam.log_conversation_end()

            bot_speaking = True
            exam.speaking = True

            try:
                await _safe_session_say(
                    reply,
                    allow_interruptions=False,
                    add_to_chat_ctx=True,
                )

                await _safe_session_say(
                    "Thank you for participating in this exam. "
                    "Could you please fill out the survey about the experiment that you received earlier? ",
                    allow_interruptions=False,
                    add_to_chat_ctx=True,
                )


            finally:
                bot_speaking = False
                exam.speaking = False

            session_running = False


            try:
                await session.aclose()
            except Exception:
                pass

            return

        # Normal non-final branch
        bot_speaking = True
        exam.speaking = True

        spoken = await _safe_session_say(
            reply,
            allow_interruptions=False,
        )

        bot_speaking = False
        exam.speaking = False
        exam.speaking_cooldown_until = time.time() + 0.2

        bot_spoken_ended_at = datetime.now(timezone.utc)

        if not spoken:
            await exam.rollback_blocked_reply(reply)
            return

        await exam.log_student_message(
            transcript,
            created_at=started_at,
            response_time_override=0.0 if force_zero_latency else student_latency_override,
        )

        last_bot_speech_end_at = bot_spoken_ended_at

        bot_latency_override = None
        if ended_at is not None:
            bot_latency_override = max(
                0.0,
                (bot_started_at - ended_at).total_seconds(),
            )

        await exam.log_bot_message(
            reply,
            created_at=bot_spoken_ended_at,
            response_time_override=bot_latency_override,
        )

    async def _flush_pending_after_window(version: int):
        nonlocal pending_transcript, merge_task, processing_task
        nonlocal pending_started_at, pending_ended_at, pending_force_zero_latency
        try:
            await asyncio.sleep(merge_window_s)
            if version != request_version or not session_running:
                return

            transcript = pending_transcript.strip()
            started_at = pending_started_at
            ended_at = pending_ended_at
            force_zero_latency = pending_force_zero_latency
            pending_transcript = ""
            pending_started_at = None
            pending_ended_at = None
            pending_force_zero_latency = False
            merge_task = None

            if not transcript:
                return

            await _cancel_processing_task()
            processing_task = asyncio.create_task(
                _process_transcript(version, transcript, started_at, ended_at, force_zero_latency)
            )
        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f"flush error: {e}")
            return

    @session.on("user_input_transcribed")
    def on_user_input(event):
        nonlocal merge_task, pending_transcript, request_version
        nonlocal inflight_transcript, inflight_version
        nonlocal pending_started_at, pending_ended_at, pending_force_zero_latency

        if exam.exam_over:
            return

        # Block mic input until the first question is spoken.
        if not accept_user_input:
            return

        if not session_running:
            return

        # Parse transcript text first so empty/noisy events don't start latency timers.
        transcript = " ".join((event.transcript or "").split()).strip()
        if not event.is_final:
            return

        now_utc = datetime.now(timezone.utc)

        if not transcript:
            return

        if pending_started_at is None:
            pending_started_at = now_utc

        if bot_speaking:
            pending_force_zero_latency = True

        # Final transcript marks student speech end for this merged turn.
        pending_ended_at = now_utc

        normalized_new = normalize_text(transcript)

        # Suppress rapid near-duplicate finals against already processed turns.
        if normalized_new and last_processed_transcript and (time.time() - last_processed_time) < 2.5:
            if is_near_duplicate(normalized_new, last_processed_transcript, max_extra_words=3):
                return

        # Suppress duplicate/refinement finals while a previous turn is pending or in-flight.
        candidates = [pending_transcript, inflight_transcript, active_transcript]
        for candidate in candidates:
            cand_norm = normalize_text(candidate)
            if is_near_duplicate(normalized_new, cand_norm, max_extra_words=3):
                return

        # Carry forward text from the superseded in-flight turn so split speech is not lost.
        if processing_task is not None and not processing_task.done():
            if inflight_transcript:
                pending_transcript = merge_transcripts(inflight_transcript, pending_transcript)
            elif active_transcript:
                pending_transcript = merge_transcripts(active_transcript, pending_transcript)

        pending_transcript = merge_transcripts(pending_transcript, transcript)
        request_version += 1

        if processing_task is not None and not processing_task.done():
            processing_task.cancel()

        if merge_task is not None and not merge_task.done():
            merge_task.cancel()

        merge_task = asyncio.create_task(_flush_pending_after_window(request_version))

    @session.on("conversation_item_added")
    def on_item_added(event):
        item = event.item
        role = getattr(item, "role", None)
        interrupted = getattr(item, "interrupted", None)
        text = getattr(item, "text_content", None)

        if interrupted and str(role or "").strip().lower() in {"assistant", "agent", "bot"}:
            asyncio.create_task(
                exam.log_system_message(
                    f"[interruption] {text or '[no text captured]'}",
                    created_at=datetime.now(timezone.utc),
                    meta={"username": exam.author_id, "interrupted": True},
                )
            )

    await exam.prewarm()

    await session.start(
        room=ctx.room,
        agent=Assistant(),
    )
    session_running = True

    bot_speaking = True
    exam.speaking = True

    await _safe_session_say(
        "Hi, I’m Acabot. Today we’ll have a short mock exam on some basic statistical concepts, including hypothesis testing, the mean and median, and variance. We’ll get started in a moment.",
        allow_interruptions=False,
        add_to_chat_ctx=False,
    )

    bot_speaking = False
    exam.speaking = False
    exam.speaking_cooldown_until = time.time() + 0.2

    # Generate and speak first question, also uninterruptible so ASR noise
    # during the first few seconds does not cut it off.
    first_q = await exam.start_exam()
    if first_q:
        print(f"\nAcabot: {first_q}\n")
        bot_speaking = True
        exam.speaking = True
        await _safe_session_say(first_q, allow_interruptions=False)
        bot_speaking = False
        exam.speaking = False
        exam.speaking_cooldown_until = time.time() + 0.2
        first_q_end = datetime.now(timezone.utc)
        last_bot_speech_end_at = first_q_end
        await exam.log_bot_message(first_q, created_at=first_q_end)

    # Open mic only after intro + first question have fully played out.
    accept_user_input = True

    await ctx.room.local_participant.publish_data(
        json.dumps({
            "type": "input_ready",
            "value": True,
        }).encode("utf-8"),
        reliable=True,
    )
    # keep worker alive until session ends
    while session_running:
        if not ctx.room.remote_participants:
            session_running = False
            break
        await asyncio.sleep(0.2)

    session_running = False
    pending_transcript = ""

    if processing_task is not None:
        try:
            await _cancel_processing_task()
        except Exception:
            pass

    if merge_task is not None and not merge_task.done():
        merge_task.cancel()

    for task in audio_recording_tasks:
        if not task.done():
            task.cancel()

    if audio_recording_tasks:
        await asyncio.gather(*audio_recording_tasks, return_exceptions=True)

    exam.logger.close_student_audio_sinks()

if __name__ == "__main__":
    # Optional helper mode to manage explicit dispatch from the same file.
    if len(sys.argv) >= 3 and sys.argv[1] == "dispatch-create":
        room_arg = sys.argv[2]
        metadata_arg = sys.argv[3] if len(sys.argv) >= 4 else "{}"
        asyncio.run(_dispatch_create(room_arg, metadata_arg))
    elif len(sys.argv) >= 3 and sys.argv[1] == "dispatch-list":
        room_arg = sys.argv[2]
        asyncio.run(_dispatch_list(room_arg))
    else:
        worker_options = dict(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            initialize_process_timeout=60,
            num_idle_processes=2,
            ws_url=LIVEKIT_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
        )
        # Only pin agent_name when explicitly configured.
        if AGENT_NAME:
            worker_options["agent_name"] = AGENT_NAME

        agents.cli.run_app(agents.WorkerOptions(**worker_options))