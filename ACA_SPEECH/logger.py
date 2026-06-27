import json
import os
import asyncio
from datetime import datetime, timezone
import csv
import wave
import re
import threading

# Append-only log file in JSONL style (one JSON object per line)
LOG_FILE = os.path.join(os.path.dirname(__file__), "platforms", "conversation_logs.json")

STUDENT_AUDIO_DIR = os.path.join(
    os.path.dirname(__file__),
    "platforms",
    "student_audio_logs"
)

class WavAudioSink:
    def __init__(self, path: str):
        self.path = path
        self.wav = None
        self.lock = threading.Lock()
        self.closed = False
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def write_frame(self, frame):
        if frame is None:
            return

        with self.lock:
            if self.closed:
                return

            if self.wav is None:
                self.wav = wave.open(self.path, "wb")
                self.wav.setnchannels(frame.num_channels)
                self.wav.setsampwidth(2)  # PCM16
                self.wav.setframerate(frame.sample_rate)

            self.wav.writeframes(bytes(frame.data))

    def close(self):
        with self.lock:
            self.closed = True

            if self.wav is not None:
                self.wav.close()
                self.wav = None

class JSONLogger:
    """
    Asynchronous JSON logger for the student-bot conversations.

    This logger records:
    - author identifiers
    - student and bot messages
    - response_time:
        * for role="user": approximate student response time in seconds
        * for role="bot" : model latency in seconds, when provided via meta["model_latency_s"]
        * for role="system" (automated messages): 0.0
    - the current rubric task associated with each message
    """

    ROLE_BY_KIND = {
        "student": "user",
        "bot": "bot",
        "feedback": "feedback",
        "system": "system",
    }

    def __init__(self, log_file: str = LOG_FILE):
        self.log_file = log_file
        self._write_lock = asyncio.Lock() # async lock to prevent concurrent appends from interleaving
        self.last_bot_timestamp = {}  # timestamp of the last bot message per conversation (used to approximate student response time)
        self._event_index = 0  # Monotonic in-process message sequence for stable ordering

        # Create log file if it does not exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write("")

        os.makedirs(STUDENT_AUDIO_DIR, exist_ok=True)
        self._student_audio_sinks = {}

    def _safe_filename_part(self, value: str) -> str:
        value = str(value or "unknown")
        value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)
        return value.strip("_") or "unknown"

    def get_student_audio_sink(self, conversation_id: str, student_id: str) -> WavAudioSink:
        safe_conversation = self._safe_filename_part(conversation_id)
        safe_student = self._safe_filename_part(student_id)

        key = (safe_conversation, safe_student)

        if key not in self._student_audio_sinks:
            path = os.path.join(
                STUDENT_AUDIO_DIR,
                safe_conversation,
                f"student_{safe_student}.wav",
            )
            self._student_audio_sinks[key] = WavAudioSink(path)

        return self._student_audio_sinks[key]

    def close_student_audio_sinks(self, conversation_id: str = None, student_id: str = None):
        if conversation_id is not None and student_id is not None:
            safe_conversation = self._safe_filename_part(conversation_id)
            safe_student = self._safe_filename_part(student_id)

            key = (safe_conversation, safe_student)

            sink = self._student_audio_sinks.get(key)
            if sink is not None:
                sink.close()

            return

        for sink in self._student_audio_sinks.values():
            sink.close()

    # Marking end of conversation
    async def end_conversation(self, conversation_id):
       """
       Explicitly mark the end of a conversation.
       Logged as a system event to preserve dialogue boundaries.
       """
       await self.log_message(
            conversation_id=conversation_id,
            role="system",
            author_id="system",
            content="[conversation ended]",
            response_time=0.0,
            created_at=None,
            rubric_task=None,
            meta={}
        )

    async def log_exam_message(
        self,
        conversation_id,
        kind,
        author_id,
        content,
        created_at,
        meta=None,
        rubric_task=None,
        response_time_override=None,
    ):
        """Single entry point for exam logging."""
        role = self.ROLE_BY_KIND.get(kind, kind)

        if kind == "student":
            await self.log_student_message(
                conversation_id=conversation_id,
                author_id=author_id,
                content=content,
                created_at=created_at,
                meta=meta,
                rubric_task=rubric_task,
                response_time_override=response_time_override,
            )
            return

        if kind == "bot":
            await self.log_bot_message(
                conversation_id=conversation_id,
                author_id=author_id,
                content=content,
                created_at=created_at,
                meta=meta,
                rubric_task=rubric_task,
                response_time_override=response_time_override,
            )
            return

        if kind == "feedback":
            await self.log_message(
                conversation_id=conversation_id,
                role="feedback",
                author_id=author_id,
                content=content,
                response_time=0.0 if response_time_override is None else float(response_time_override),
                created_at=created_at,
                rubric_task=rubric_task,
                meta=meta,
            )
            return

        await self.log_message(
            conversation_id=conversation_id,
            role=role,
            author_id=author_id,
            content=content,
            response_time=0.0 if response_time_override is None else float(response_time_override),
            created_at=created_at,
            rubric_task=rubric_task,
            meta=meta,
        )

    # Student Logging
    async def log_student_message(
        self,
        conversation_id,
        author_id,
        content,
        created_at,
        meta=None,
        rubric_task=None,
        response_time_override=None,
    ):
        """
        Log a student message.
        response_time is approximated as the time between the last logged bot message in the conversation and the time of the student response in seconds.
        When response_time_override is provided, it is used directly.
        """

        if response_time_override is not None:
            latency = float(response_time_override)
        else:
            latency = 0.0

            # Use Discord timestamps (created_at) to approximate student response time
            if conversation_id in self.last_bot_timestamp and created_at is not None:
                try:
                    latency = (created_at - self.last_bot_timestamp[conversation_id]).total_seconds()
                    # Set extreme idle times (> 5 minutes) and negative values to 0.0
                    if latency > 300 or latency < 0: # <0 happens when the student answers before the next question has popped up
                        latency = 0.0
                except Exception:
                    latency = 0.0

        await self.log_message(
            conversation_id=conversation_id,
            role="user",
            author_id=author_id,
            content=content,
            response_time=latency,
            created_at=created_at,
            rubric_task=rubric_task,
            meta=meta
        )

    # Bot logging
    async def log_bot_message(
        self,
        conversation_id,
        author_id,
        content,
        created_at,
        meta=None,
        rubric_task=None,
        response_time_override=None,
    ):
        # response_time for bot messages is explicit override when provided; otherwise fallback to model latency.
        if response_time_override is not None:
            response_time = float(response_time_override)
        else:
            response_time = (round(meta["model_latency_s"], 3) if meta and "model_latency_s" in meta else 0.0)

        # Store the timestamp of the last bot speech end per conversation.
        if created_at is not None:
            self.last_bot_timestamp[conversation_id] = created_at

        await self.log_message(
            conversation_id=conversation_id,
            role="bot",
            author_id=author_id,
            content=content,
            response_time=response_time,
            created_at=created_at,
            meta=meta,
            rubric_task=rubric_task,
        )

    # System logging
    async def log_system_message(
        self,
        conversation_id,
        author_id,
        content,
        created_at=None,
        meta=None,
        rubric_task=None,
        response_time_override=None,
    ):
        await self.log_message(
            conversation_id=conversation_id,
            role="system",
            author_id=author_id,
            content=content,
            response_time=0.0 if response_time_override is None else float(response_time_override),
            created_at=created_at,
            rubric_task=rubric_task,
            meta=meta,
        )

    # Core Logging
    async def log_message(self, conversation_id, role, author_id, content, response_time, created_at=None, meta=None, rubric_task=None):
        # Prefer Discord timestamp (when available) for later ordering; fallback to server timestamp
        discord_ts = None
        if created_at is not None:
            try:
                # Discord gives an aware datetime; store ISO UTC for exact ordering later
                discord_ts = created_at.astimezone(timezone.utc).isoformat()
            except Exception:
                discord_ts = None

        # Assign monotonic sequence under the same lock used for append.
        self._event_index += 1
        event_index = self._event_index

        log_entry = {
            "event_index": event_index,
            "timestamp_server_utc": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),  # time of logging (UTC)
            "timestamp_discord_utc": discord_ts,  # original Discord message time (UTC); may be None for system events
            "conversation_id": conversation_id,
            "role": role,  # "user",  "bot", "feedback" or "system"
            "author_id": author_id,
            "content": content,
            "response_time": response_time,
            "rubric_task": rubric_task,
            "meta": meta or {}
        }

        # Use a lock to prevent concurrent appends from interleaving
        async with self._write_lock:
            await asyncio.to_thread(self.append_log, log_entry)

    async def log_feedback(
            self,
            conversation_id,
            author_id,
            content,
            created_at=None,
            meta=None,
            rubric_task=None,
    ):
        """Log feedback message at end of exam."""
        await self.log_message(
            conversation_id=conversation_id,
            role="feedback",
            author_id=author_id,
            content=content,
            response_time=0.0,
            created_at=created_at or datetime.now(timezone.utc),
            rubric_task=rubric_task,
            meta=meta,
        )

    async def log_feedback_message(
            self,
            conversation_id,
            author_id,
            content,
            created_at=None,
            meta=None,
            rubric_task=None,
    ):
        """Compatibility wrapper for modes that call log_feedback_message()."""
        await self.log_feedback(
            conversation_id=conversation_id,
            author_id=author_id,
            content=content,
            created_at=created_at,
            meta=meta,
            rubric_task=rubric_task,
        )

    def append_log(self, entry):
        # Append as one JSON object per line (JSONL format)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


    # Reading and grouping logs
    async def read_logs_grouped(self):
        """Return logs grouped by conversation_id, sorted by deterministic sequence then timestamp."""
        with open(self.log_file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]

        def sort_key(x):
            idx = x.get("event_index")
            if isinstance(idx, int):
                return (0, idx)
            return (1, x.get("timestamp_discord_utc") or x.get("timestamp_server_utc") or "")

        data.sort(key=sort_key)

        grouped = {}
        for entry in data:
            conv = entry["conversation_id"]
            grouped.setdefault(conv, []).append(entry)
        return grouped

    # Export log per user
    async def export_conversation_csvs(self, export_dir="exports"):
        """
        Export one CSV per conversation_id.
        The filename uses meta["username"] from the first user message that contains it and otherwise "unknown_user".
        Each CSV contains a role ("user" or "bot"), messages, response_time and rubric_task column.
        """
        grouped = await self.read_logs_grouped()
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)

        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        for conv_id, messages in grouped.items():
            if not messages:
                continue

            # Sort by deterministic sequence first, then timestamps as fallback.
            messages.sort(
                key=lambda x: (
                    0 if isinstance(x.get("event_index"), int) else 1,
                    x.get("event_index") if isinstance(x.get("event_index"), int) else (x.get("timestamp_discord_utc") or x.get("timestamp_server_utc") or "")
                )
            )

            # Read username from meta of first user message that includes "username"
            username = "unknown_user"
            for msg in messages:
                if msg["role"] == "user":
                    meta = msg.get("meta") or {}
                    if "username" in meta:
                        username = meta["username"]
                        break

            filename = f"{username}_{now_str}.csv"

            # Collect dialogue rows
            rows = []
            for msg in messages:
                if msg["role"] in ["user", "bot", "feedback"]:
                    rows.append({
                        "role": msg["role"],
                        "content": msg["content"],
                        "response_time": msg["response_time"],
                        "rubric_task": msg.get("rubric_task"),
                        "debug_state": json.dumps(
                            (msg.get("meta") or {}).get("debug_state"),
                            ensure_ascii=False
                        ) if msg["role"] == "user" else ""
                    })

            if not rows:
                continue

            path = os.path.join(export_dir, filename)

            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["role", "content", "response_time", "rubric_task", "debug_state"])
                writer.writeheader()
                writer.writerows(rows)

            print(f"Exported {path}")

class ExamLogMixin:
    log_mode = "unknown"
    _log_timeout_s = 1.5

    async def _current_rubric_task(self):
        current_question = getattr(self, "dialog", None)
        if current_question is None:
            return None

        current_question = getattr(self.dialog, "current_question", None)
        if current_question is None:
            return None

        if isinstance(current_question, dict):
            task = current_question.get("Task")
            return str(task).strip() if task is not None and str(task).strip() else None

        try:
            if hasattr(current_question, "get"):
                task = current_question.get("Task")
            else:
                task = current_question["Task"]
            return str(task).strip() if task is not None and str(task).strip() else None
        except Exception:
            return None

    async def log_bot_message(self, content, *, created_at=None, response_time_override=None):
        await self.logger.log_bot_message(
            conversation_id=self.conversation_id,
            author_id="acabot",
            content=content,
            created_at=created_at,
            rubric_task=await self._current_rubric_task(),
            meta={
                "username": self.author_id,
                "model_latency_s": response_time_override if response_time_override is not None else getattr(self, "last_model_latency_s", 0.0),
            },
        )

    async def log_student_message(
            self,
            content,
            *,
            created_at=None,
            response_time_override=None,
            debug_state_override=None,
            rubric_task_override=None,
            current_subtopic_override=None,
    ):
        debug_state = debug_state_override
        rubric_task = rubric_task_override

        if debug_state is None and getattr(self, "dialog", None) is not None:
            debug_state = getattr(self.dialog, "last_debug_state", None)

        if rubric_task is None:
            rubric_task = await self._current_rubric_task()

        meta = {
            "username": self.author_id,
            "response_time_s": response_time_override,
            "debug_state": debug_state,
        }

        if getattr(self, "log_mode", None) == "simulation":
            synthetic_student = getattr(self, "synthetic_student", None)

            if synthetic_student is not None and hasattr(
                    synthetic_student,
                    "get_sampled_knowledge_state",
            ):
                if current_subtopic_override is not None:
                    current_subtopic = str(current_subtopic_override or "").strip()
                elif getattr(self, "dialog", None) is not None:
                    current_subtopic = str(
                        getattr(self.dialog, "current_target_subtopic", "") or ""
                    ).strip()
                else:
                    current_subtopic = ""

                meta["simulation_knowledge_state"] = (
                    synthetic_student.get_sampled_knowledge_state(
                        current_subtopic=current_subtopic
                    )
                )

        await self.logger.log_student_message(
            conversation_id=self.conversation_id,
            author_id=self.author_id,
            content=content,
            created_at=created_at,
            rubric_task=rubric_task,
            meta=meta,
            response_time_override=response_time_override,
        )
    async def log_feedback_message(self, content, *, created_at=None):
        await self.logger.log_feedback(
            conversation_id=self.conversation_id,
            author_id="acabot",
            content=content,
            created_at=created_at,
            rubric_task=await self._current_rubric_task(),
            meta={"username": self.author_id},
        )

    async def log_system_message(self, content, *, created_at=None, meta=None):
        await self.logger.log_system_message(
            conversation_id=self.conversation_id,
            author_id="system",
            content=content,
            created_at=created_at,
            rubric_task=await self._current_rubric_task(),
            meta=meta or {"username": self.author_id},
        )

    async def log_conversation_end(self):
        await self.logger.end_conversation(self.conversation_id)