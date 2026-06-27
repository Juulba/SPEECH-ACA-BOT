import argparse
import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import csv

from dialog_instance import Dialog
from logger import JSONLogger, ExamLogMixin
from setup.constants import document, rubrics
from synthetic_student import SyntheticStudent


class SimulatedExamSession(ExamLogMixin):
    """Runs a fully simulated examiner-student oral exam."""

    def __init__(
        self,
        author_id: str = "llm_student",
        student_profile: Optional[dict] = None,
        save_transcript: bool = True,
    ) -> None:
        self.log_mode = "simulation"
        self.dialog: Optional[Dialog] = None
        self._dialog_init_lock = asyncio.Lock()

        self.logger = JSONLogger()
        self.conversation_id = str(uuid.uuid4())
        self.author_id = author_id
        self.exam_over = False
        self.last_model_latency_s = 0.0

        self.student_profile = student_profile or {
            "knowledge_state_probs": {
                "correct": 0.35,
                "partial": 0.40,
                "wrong": 0.15,
                "no_knowledge": 0.10,
            },
          }

        self.synthetic_student = SyntheticStudent(profile=self.student_profile)

        self.save_transcript = save_transcript
        self.transcript = []

    async def _ensure_dialog(self) -> Dialog:
        if self.dialog is not None:
            return self.dialog

        async with self._dialog_init_lock:
            if self.dialog is None:
                self.dialog = await asyncio.to_thread(
                    Dialog,
                    document,
                    rubrics,
                )

        return self.dialog

    @staticmethod
    def _get_current_task_context_and_subtopic(dialog: Dialog) -> tuple[str, str, str]:
        if dialog.current_question is None:
            return "", "", ""

        try:
            current_task = str(dialog.current_question.get("Task", "")).strip()
        except Exception:
            current_task = str(dialog.current_question["Task"]).strip()

        current_subtopic = str(
            getattr(dialog, "current_target_subtopic", "") or ""
        ).strip()

        try:
            current_context = dialog.get_student_context()
        except AttributeError:
            current_context = dialog._get_task_window()

        return current_task, current_context, current_subtopic

    @staticmethod
    def _strip_exam_over_marker(message: str) -> tuple[str, bool]:
        if "[examOver]" not in message:
            return message, False

        cleaned = message.replace("[examOver]", "").strip()
        return cleaned, True

    async def run(self, max_turns: int = 50) -> list[dict]:
        dialog = await self._ensure_dialog()

        start = asyncio.get_running_loop().time()
        examiner_message = (await dialog.send_message("")) or ""
        self.last_model_latency_s = max(
            0.0,
            asyncio.get_running_loop().time() - start,
        )

        examiner_message, exam_over = self._strip_exam_over_marker(examiner_message)
        self.exam_over = exam_over

        first_q_time = datetime.now(timezone.utc)
        self.transcript.append({"role": "examiner", "text": examiner_message})
        print(f"Examiner: {examiner_message}\n")

        if self.exam_over:
            await self.log_feedback_message(examiner_message, created_at=first_q_time)
            return self.transcript

        await self.log_bot_message(examiner_message, created_at=first_q_time)

        turns = 0

        while turns < max_turns and not self.exam_over:
            current_task, current_context, current_subtopic = (
                self._get_current_task_context_and_subtopic(dialog)
            )

            start = asyncio.get_running_loop().time()

            student_message = await self.synthetic_student.answer(
                examiner_message=examiner_message,
                current_task=current_task,
                current_subtopic=current_subtopic,
                current_context=current_context,
            )

            self.last_model_latency_s = max(
                0.0,
                asyncio.get_running_loop().time() - start,
            )

            student_time = datetime.now(timezone.utc)
            knowledge_state = self.synthetic_student.get_sampled_knowledge_state(
                current_subtopic=current_subtopic
            )

            self.transcript.append({
                "role": "student",
                "text": student_message,
                "current_task": current_task,
                "current_subtopic": current_subtopic,
                "sampled_knowledge_state": knowledge_state.get("current_state"),
            })

            print(f"Student: {student_message}\n")

            start = asyncio.get_running_loop().time()
            examiner_reply = (await dialog.send_message(student_message)) or ""
            self.last_model_latency_s = max(
                0.0,
                asyncio.get_running_loop().time() - start,
            )

            latest_debug_state = getattr(dialog, "last_debug_state", None)

            await self.log_student_message(
                student_message,
                created_at=student_time,
                debug_state_override=latest_debug_state,
                rubric_task_override=current_task,
                current_subtopic_override=current_subtopic,
            )

            examiner_reply, exam_over = self._strip_exam_over_marker(examiner_reply)
            self.exam_over = exam_over

            examiner_time = datetime.now(timezone.utc)
            self.transcript.append({"role": "examiner", "text": examiner_reply})
            print(f"Examiner: {examiner_reply}\n")

            if self.exam_over:
                await self.log_feedback_message(examiner_reply, created_at=examiner_time)
            else:
                await self.log_bot_message(examiner_reply, created_at=examiner_time)

            examiner_message = examiner_reply
            turns += 1

        return self.transcript

    def save_transcript_to_file(self, output_path: Optional[str] = None) -> Optional[Path]:
        if not self.save_transcript:
            return None

        out_dir = Path("simulated_runs")
        out_dir.mkdir(parents=True, exist_ok=True)

        if output_path:
            output_file = Path(output_path)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = out_dir / f"simulated_exam_{timestamp}.json"

        payload = {
            "conversation_id": self.conversation_id,
            "author_id": self.author_id,
            "student_profile": self.student_profile,
            "synthetic_student_tokens": self.synthetic_student.tokens,
            "examiner_tokens": getattr(self.dialog, "tokens", None),
            "sampled_subtopic_states": dict(self.synthetic_student.sampled_subtopic_states),
            "exam_over": self.exam_over,
            "transcript": self.transcript,
        }

        output_file.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return output_file

    def save_knowledge_profile_csv(
            self,
            run_id: int,
            output_path: str = "simulated_runs/knowledge_profiles.csv",
    ) -> Path:
        """
        Append this simulated student's full knowledge profile to a CSV.

        One row = one student-subtopic knowledge state.
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        file_exists = output_file.exists()

        with output_file.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "run_id",
                    "conversation_id",
                    "student_id",
                    "subtopic",
                    "knowledge_state",
                ],
            )

            if not file_exists:
                writer.writeheader()

            for subtopic, state in self.synthetic_student.sampled_subtopic_states.items():
                writer.writerow({
                    "run_id": run_id,
                    "conversation_id": self.conversation_id,
                    "student_id": self.author_id,
                    "subtopic": subtopic,
                    "knowledge_state": state,
                })

        return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fully simulated Acabot oral exam."
    )

    parser.add_argument(
        "--user",
        default="llm_student",
        help="Username stored in logs.",
    )

    parser.add_argument(
        "--max-turns",
        type=int,
        default=50,
        help="Maximum number of simulated student turns.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON file path for transcript output.",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of simulated exams to run",
    )

    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    student_profile = {
        "knowledge_state_probs": {
            "correct": 0.35,
            "partial": 0.40,
            "wrong": 0.15,
            "no_knowledge": 0.10,
        },
    }

    all_runs = []

    for i in range(args.runs):
        print(f"\n--- RUN {i + 1}/{args.runs} ---\n")

        session = SimulatedExamSession(
            author_id=f"{args.user}_{i}",
            student_profile=student_profile,
            save_transcript=True,
        )

        try:
            transcript = await session.run(max_turns=args.max_turns)

            output_path = args.output
            if args.output and args.runs > 1:
                base = Path(args.output)
                output_path = str(base.with_name(f"{base.stem}_{i + 1}{base.suffix}"))

            saved = session.save_transcript_to_file(output_path)
            profile_csv = session.save_knowledge_profile_csv(run_id=i)

            all_runs.append({
                "run_id": i,
                "conversation_id": session.conversation_id,
                "exam_over": session.exam_over,
                "transcript": transcript,
            })

            if session.exam_over:
                print("Simulated exam finished.")
            else:
                print("Simulation stopped due to max_turns.")

            if saved:
                print(f"Transcript saved to: {saved}")

            if profile_csv:
                print(f"Knowledge profile saved to: {profile_csv}")

        finally:
            await session.log_conversation_end()

    print(f"\nCompleted {len(all_runs)} runs.")


if __name__ == "__main__":
    asyncio.run(main())