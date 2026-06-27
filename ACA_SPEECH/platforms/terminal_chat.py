import argparse
import asyncio
import uuid
from datetime import datetime, timezone
from typing import Optional

from dialog_instance import Dialog
from logger import JSONLogger, ExamLogMixin
from setup.constants import document, rubrics


class ChatExamSession(ExamLogMixin):
    """Text-only exam session for local testing without voice/LiveKit."""

    def __init__(self, author_id: str = "chat_test_user") -> None:
        self.dialog: Optional[Dialog] = None
        self._dialog_init_lock = asyncio.Lock()
        self.logger = JSONLogger()
        self.conversation_id = str(uuid.uuid4())
        self.author_id = author_id
        self.exam_over = False
        self.last_model_latency_s = 0.0

    async def _ensure_dialog(self) -> Dialog:
        if self.dialog is not None:
            return self.dialog

        async with self._dialog_init_lock:
            if self.dialog is None:
                self.dialog = await asyncio.to_thread(Dialog, document, rubrics)

        return self.dialog

    async def start_exam(self) -> str:
        dialog = await self._ensure_dialog()
        start = asyncio.get_running_loop().time()
        first_q = (await dialog.send_message("")) or ""
        self.last_model_latency_s = max(0.0, asyncio.get_running_loop().time() - start)
        return first_q

    async def handle_user_turn(self, transcript: str) -> str:
        if self.exam_over:
            return ""

        text = " ".join((transcript or "").split()).strip()
        if not text:
            return ""

        dialog = await self._ensure_dialog()
        start = asyncio.get_running_loop().time()
        reply = (await dialog.send_message(text)) or ""
        self.last_model_latency_s = max(0.0, asyncio.get_running_loop().time() - start)

        if "[examOver]" in reply:
            self.exam_over = True
            reply = reply.replace("[examOver]", "").strip()

        return reply


async def run_terminal_chat(author_id: str) -> None:
    exam = ChatExamSession(author_id=author_id)

    print("Acabot terminal chat test")
    print("Type your answer and press Enter.")
    print("Use '/exit' to stop.\n")

    first_q = await exam.start_exam()
    if first_q:
        first_q_time = datetime.now(timezone.utc)
        print(f"Acabot: {first_q}\n")
        await exam.log_bot_message(first_q, created_at=first_q_time)

    try:
        while not exam.exam_over:
            user_text = await asyncio.to_thread(input, "You: ")
            user_text = " ".join((user_text or "").split()).strip()

            if not user_text:
                continue

            if user_text.lower() in {"/exit", "exit", "quit", "/quit"}:
                break

            user_time = datetime.now(timezone.utc)
            reply = await exam.handle_user_turn(user_text)

            await exam.log_student_message(user_text, created_at=user_time)

            if reply:
                bot_time = datetime.now(timezone.utc)

                # Check if this is feedback (contains exam end marker)
                is_feedback = "[examOver]" in reply
                reply_clean = reply.replace("[examOver]", "").strip()

                print(f"Acabot: {reply_clean}\n")

                if is_feedback:
                    await exam.log_feedback_message(reply_clean, created_at=bot_time)
                else:
                    await exam.log_bot_message(reply_clean, created_at=bot_time)

        if exam.exam_over:
            print("Exam finished.")
        else:
            print("Chat ended by user.")
    finally:
        await exam.log_conversation_end()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Acabot in terminal text mode for testing.")
    parser.add_argument("--user", default="chat_test_user", help="Username stored in conversation logs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_terminal_chat(author_id=args.user))
