"""
Created on Tue Oct  1 16:21:21 2024

See LICENSE file in the root of the repository.

Copyright (c) Julie Bauer and Aki Härmä, DACS/FSE, Maastricht University, 2026
"""

import json
import asyncio

from pdf_processor import PDFProcessor
from rubric import Rubric
from utils import clean_string, hard_cap_words
from LLM.llm_calls import make_call, make_simple_call
from setup.constants import *
from vector_db import VectorDB
import copy


GLOBAL_VECTORDB = None

def get_global_vector_db():
    global GLOBAL_VECTORDB
    if GLOBAL_VECTORDB is None:
        GLOBAL_VECTORDB = VectorDB(clear_on_init=False)
    return GLOBAL_VECTORDB

class Dialog:
    """
    A class for one dialogue.
    The controller uses one state call per student turn.
    The state call decides both reasoning state and subtopic coverage.
    """

    def __init__(self, document, rubric, vector_db=None):
        self.first_message = False
        self.is_first_task = True

        self.pdf_processor = PDFProcessor()

        self.task_history = []
        self.full_history = []

        text_segments = self.pdf_processor.load_pdf(document)

        # Use the prewarmed vector DB if one is provided.
        # Otherwise fall back to the normal global vector DB.
        self.vector_db = vector_db or get_global_vector_db()
        self.vector_db.ensure_segments(text_segments)

        self.rubric = Rubric(rubric)
        self.rubric.df["Context"] = self.rubric.df["Task"].apply(
            lambda task: "\n".join(self.vector_db.query_db(task, top_k=2))
        )

        self.current_question = self.rubric.get_next_question()
        self.tokens = 0

        # State/controller
        self.current_state = None
        self.current_behavior = "OPEN"
        self.current_target_subtopic = ""

        # Control limits
        self.max_student_turns_per_task = 30

        # Debug
        self.state_history = []
        self.debug_states = True

        # Ordered subtopic targeting
        self.completed_target_subtopics = set()
        self.target_opened = set()

        # Persistent diagnostic memory for the current rubric task
        self.diagnostically_resolved_memory = set()
        self.partial_subtopic_memory = set()

        # follow-ups
        self.target_followup_count = {}
        self.max_followups_per_subtopic = 2

        # Silence tracking for non-diagnostic responses
        self.silence_count_for_target = {}
        self.max_silences_before_non_diagnostic = 1

        self.last_debug_state = None

    def snapshot_state(self) -> dict:
        return {
            "first_message": self.first_message,
            "is_first_task": self.is_first_task,
            "task_history": list(self.task_history),
            "full_history": list(self.full_history),
            "current_question": copy.deepcopy(self.current_question),
            "tokens": self.tokens,
            "current_state": copy.deepcopy(self.current_state),
            "current_behavior": self.current_behavior,
            "current_target_subtopic": self.current_target_subtopic,
            "state_history": copy.deepcopy(self.state_history),
            "completed_target_subtopics": set(self.completed_target_subtopics),
            "target_opened": set(self.target_opened),
            "target_followup_count": dict(self.target_followup_count),
            "diagnostically_resolved_memory": set(self.diagnostically_resolved_memory),
            "partial_subtopic_memory": set(self.partial_subtopic_memory),
            "rubric_df": self.rubric.df.copy(deep=True),
            "silence_count_for_target": dict(self.silence_count_for_target),
        }

    def restore_state(self, snapshot: dict) -> None:
        self.first_message = snapshot["first_message"]
        self.is_first_task = snapshot["is_first_task"]
        self.task_history = list(snapshot["task_history"])
        self.full_history = list(snapshot["full_history"])
        self.current_question = copy.deepcopy(snapshot["current_question"])
        self.tokens = snapshot["tokens"]
        self.current_state = copy.deepcopy(snapshot["current_state"])
        self.current_behavior = snapshot["current_behavior"]
        self.current_target_subtopic = snapshot["current_target_subtopic"]
        self.state_history = copy.deepcopy(snapshot["state_history"])
        self.completed_target_subtopics = set(snapshot["completed_target_subtopics"])
        self.target_opened = set(snapshot["target_opened"])
        self.target_followup_count = dict(snapshot["target_followup_count"])
        self.diagnostically_resolved_memory = set(
            snapshot["diagnostically_resolved_memory"]
        )
        self.partial_subtopic_memory = set(snapshot["partial_subtopic_memory"])
        self.rubric.df = snapshot["rubric_df"].copy(deep=True)
        self.silence_count_for_target = dict(
            snapshot.get("silence_count_for_target", {})
        )

    def _silences_used_for_target(self, subtopic: str) -> int:
        key = self._target_key(subtopic)
        return self.silence_count_for_target.get(key, 0)

    def _increment_silence_for_target(self, subtopic: str) -> None:
        key = self._target_key(subtopic)

        if key:
            self.silence_count_for_target[key] = (
                    self.silence_count_for_target.get(key, 0) + 1
            )

    def _reset_silence_for_target(self, subtopic: str) -> None:
        key = self._target_key(subtopic)

        if key and key in self.silence_count_for_target:
            self.silence_count_for_target.pop(key)

    def _followups_used_for_target(self, subtopic: str) -> int:
        key = self._target_key(subtopic)
        return self.target_followup_count.get(key, 0)

    def _increment_followup_for_target(self, subtopic: str) -> None:
        key = self._target_key(subtopic)

        if key:
            self.target_followup_count[key] = (
                    self.target_followup_count.get(key, 0) + 1
            )

    def _apply_diagnostic_memory(self, state: dict) -> dict:
        """
        Make diagnostic coverage persistent within the current task.

        Rules:
        - Once resolved, always resolved.
        - Once partial, stays partial unless later resolved.
        - A partial topic may not later become non_diagnostic_response.
        - non_diagnostic_response is local to the current target only.
        - unseen may become non_diagnostic_response only for the current target.
        """

        topics = self._get_current_topics_list()
        topic_norm_to_original = {
            self._normalize_subtopic(topic): topic
            for topic in topics
        }

        current_resolved = self._norm_set(
            state.get("diagnostically_resolved", [])
        )
        current_partial = self._norm_set(
            state.get("partially_reasoned", [])
        )
        current_non_diagnostic = self._norm_set(
            state.get("non_diagnostic_response", [])
        )
        current_unseen = self._norm_set(
            state.get("unseen_subtopics", [])
        )

        valid_topics = set(topic_norm_to_original.keys())
        current_target_norm = self._normalize_subtopic(
            self.current_target_subtopic
        )

        # Keep only valid rubric topics.
        current_resolved &= valid_topics
        current_partial &= valid_topics
        current_non_diagnostic &= valid_topics
        current_unseen &= valid_topics

        # Only the current target may be non_diagnostic_response.
        current_non_diagnostic = {
            norm for norm in current_non_diagnostic
            if norm == current_target_norm
        }

        # Store newly resolved topics.
        for norm in current_resolved:
            self.diagnostically_resolved_memory.add(norm)

        # Store newly partial topics, unless already resolved.
        for norm in current_partial:
            if norm not in self.diagnostically_resolved_memory:
                self.partial_subtopic_memory.add(norm)

        # Resolved beats partial.
        self.partial_subtopic_memory -= self.diagnostically_resolved_memory

        resolved = set(self.diagnostically_resolved_memory)
        partial = set(self.partial_subtopic_memory)

        # Memory is reapplied.
        current_resolved |= resolved
        current_partial |= partial

        # Priority:
        # resolved > partial > non_diagnostic > unseen
        current_partial -= current_resolved

        # Important rule:
        # partial cannot become non_diagnostic_response.
        current_non_diagnostic -= current_resolved
        current_non_diagnostic -= current_partial

        current_unseen -= current_resolved
        current_unseen -= current_partial
        current_unseen -= current_non_diagnostic

        # Any missing valid topics become unseen.
        mentioned = (
                current_resolved
                | current_partial
                | current_non_diagnostic
                | current_unseen
        )
        current_unseen |= valid_topics - mentioned

        state["diagnostically_resolved"] = [
            topic
            for norm, topic in topic_norm_to_original.items()
            if norm in current_resolved
        ]

        state["partially_reasoned"] = [
            topic
            for norm, topic in topic_norm_to_original.items()
            if norm in current_partial
        ]

        state["non_diagnostic_response"] = [
            topic
            for norm, topic in topic_norm_to_original.items()
            if norm in current_non_diagnostic
        ]

        state["unseen_subtopics"] = [
            topic
            for norm, topic in topic_norm_to_original.items()
            if norm in current_unseen
        ]

        return state

    def _target_key(self, subtopic: str) -> str:
        return self._normalize_subtopic(subtopic)

    def _next_ordered_target_subtopic(self) -> str:
        topics = self._get_current_topics_list()

        for topic in topics:
            key = self._target_key(topic)

            if not key:
                continue

            if key in self.completed_target_subtopics:
                continue

            if key in self.diagnostically_resolved_memory:
                continue

            return topic

        return ""

    def _mark_target_done(self, subtopic: str) -> None:
        key = self._target_key(subtopic)

        if key:
            self.completed_target_subtopics.add(key)

    def _target_status(self, target: str, state: dict) -> str:
        target_norm = self._normalize_subtopic(target)

        if not target_norm:
            return "none"

        resolved = self._norm_set(state.get("diagnostically_resolved", []))
        partial = self._norm_set(state.get("partially_reasoned", []))
        non_diagnostic = self._norm_set(state.get("non_diagnostic_response", []))
        unseen = self._norm_set(state.get("unseen_subtopics", []))

        if target_norm in resolved:
            return "resolved"

        if target_norm in partial:
            return "partially_reasoned"

        if target_norm in non_diagnostic:
            return "non_diagnostic_response"

        if target_norm in unseen:
            return "unseen"

        return "unknown"

    def _target_has_been_opened(self, subtopic: str) -> bool:
        key = self._target_key(subtopic)
        return key in self.target_opened

    def _mark_target_opened(self, subtopic: str) -> None:
        key = self._target_key(subtopic)

        if key:
            self.target_opened.add(key)

    def _enforce_controller_invariants(self, state: dict) -> None:
        target = self.current_target_subtopic
        target_status = self._target_status(target, state)

        if not target:
            return

        if self.current_behavior == "FOLLOW_UP" and target_status == "resolved":
            raise RuntimeError(
                "Controller invariant violated: FOLLOW_UP was assigned to a "
                f"resolved target: {target}"
            )

        if self.current_behavior == "EXPLORE_NEW" and target_status == "resolved":
            raise RuntimeError(
                "Controller invariant violated: EXPLORE_NEW was assigned to a "
                f"resolved target: {target}"
            )

    def _repair_diagnostic_state(self, state: dict) -> dict:
        topics = self._get_current_topics_list()
        valid = {self._normalize_subtopic(t): t for t in topics}

        resolved = self._norm_set(state.get("diagnostically_resolved", []))
        partial = self._norm_set(state.get("partially_reasoned", []))
        non_diagnostic = self._norm_set(state.get("non_diagnostic_response", []))
        unseen = self._norm_set(state.get("unseen_subtopics", []))

        resolved &= valid.keys()
        partial &= valid.keys()
        non_diagnostic &= valid.keys()
        unseen &= valid.keys()

        # ------------------------------------------------------------
        # Guardrail:
        # A topic can only be non_diagnostic_response if it is the
        # current target subtopic.
        #
        # This prevents the LLM from closing future, previous, or
        # non-current subtopics.
        # ------------------------------------------------------------
        current_target_norm = self._normalize_subtopic(self.current_target_subtopic)

        invalid_non_diagnostic = {
            norm for norm in non_diagnostic
            if norm != current_target_norm
        }

        non_diagnostic -= invalid_non_diagnostic
        unseen |= invalid_non_diagnostic

        # ------------------------------------------------------------
        # Priority:
        # resolved > partial > non_diagnostic > unseen
        # ------------------------------------------------------------
        partial -= resolved

        non_diagnostic -= resolved
        non_diagnostic -= partial

        unseen -= resolved
        unseen -= partial
        unseen -= non_diagnostic

        mentioned = resolved | partial | non_diagnostic | unseen
        missing = set(valid.keys()) - mentioned
        unseen |= missing

        state["diagnostically_resolved"] = [
            topic for norm, topic in valid.items() if norm in resolved
        ]

        state["partially_reasoned"] = [
            topic for norm, topic in valid.items() if norm in partial
        ]

        state["non_diagnostic_response"] = [
            topic for norm, topic in valid.items() if norm in non_diagnostic
        ]

        state["unseen_subtopics"] = [
            topic for norm, topic in valid.items() if norm in unseen
        ]

        return state

    def _get_task_window(self) -> str:
        return "\n".join(self.task_history).strip()

    def get_student_context(self) -> str:
        """
        Return the current task-level conversation history for the synthetic student.
        This gives enough context for follow-up answers without including previous tasks.
        """
        return self._get_task_window()

    def _get_full_window(self) -> str:
        return "\n".join(self.full_history).strip()

    def _normalize_subtopic(self, value: str) -> str:
        return str(value or "").strip().lower()

    def _norm_set(self, items) -> set:
        return {
            self._normalize_subtopic(item)
            for item in (items or [])
            if self._normalize_subtopic(item)
        }

    def _append_turn(self, speaker: str, text: str) -> None:
        turn = f"{speaker}: {text}"
        self.task_history.append(turn)
        self.full_history.append(turn)

    def _student_turn_count_current_task(self) -> int:
        return sum(
            1 for turn in self.task_history
            if turn.startswith("Student:")
        )

    def _get_current_topics_list(self) -> list:
        current_topics = self.rubric.get_topics_for_question(
            self.current_question.name
        )

        if current_topics is None:
            return []

        if isinstance(current_topics, str):
            return [
                line.strip("-• \t")
                for line in current_topics.split("\n")
                if line.strip()
            ]

        try:
            return [
                str(topic).strip()
                for topic in current_topics
                if str(topic).strip()
            ]
        except Exception:
            return [str(current_topics).strip()]

    async def _measure_state(self) -> dict:
        """
        Single state call:
        - classifies diagnostic evidence for each fixed rubric subtopic
        - places every subtopic into exactly one diagnostic category:
            diagnostically_resolved,
            partially_reasoned,
            non_diagnostic_response,
            unseen_subtopics
        - does NOT decide the next subtopic
        - does NOT decide examiner behavior
        """

        task_window = self._get_task_window()

        current_target = (
                self.current_target_subtopic
                or self._next_ordered_target_subtopic()
        )

        current_topics = self._get_current_topics_list()

        state_prompt_filled = prompts["state_prompt"].format(
            self.current_question["Task"],
            current_target,
            current_topics,
            task_window,
        )

        result = await make_simple_call(
            state_prompt_filled,
            MODEL,
            max_tokens=512,
        )

        raw = result[0] if isinstance(result, (list, tuple)) else result

        if isinstance(result, (list, tuple)) and len(result) > 1:
            self.tokens += result[1]

        try:
            state = json.loads(raw)
        except Exception:
            state = {
                "diagnostically_resolved": [],
                "partially_reasoned": [],
                "non_diagnostic_response": [],
                "unseen_subtopics": current_topics,
                "reason": "Could not parse model state JSON.",
            }

        # Normalize required fields.
        state["diagnostically_resolved"] = (
                state.get("diagnostically_resolved") or []
        )
        state["partially_reasoned"] = (
                state.get("partially_reasoned") or []
        )
        state["non_diagnostic_response"] = (
                state.get("non_diagnostic_response") or []
        )
        state["unseen_subtopics"] = (
                state.get("unseen_subtopics") or []
        )
        state["reason"] = state.get("reason") or ""

        state = self._repair_diagnostic_state(state)
        state = self._apply_diagnostic_memory(state)

        return state

    def _update_state_controls(self, state: dict) -> str:
        """
        Controller decision based only on diagnostic subtopic status.

        Diagnostic statuses:
        - unseen
        - partially_reasoned
        - diagnostically_resolved
        - non_diagnostic_response

        Controller behaviors:
        - OPEN
        - EXPLORE_NEW
        - FOLLOW_UP

        A non_diagnostic_response means the current target was explicitly attempted,
        but the student's response gave no meaningful usable evidence, such as
        "I don't know", silence, refusal, random content, or irrelevant content.

        For dialogue control, non_diagnostic_response closes the target and moves on.
        """

        self.current_state = state

        target = self.current_target_subtopic or self._next_ordered_target_subtopic()

        if not target:
            self.current_behavior = "OPEN"
            self.current_target_subtopic = ""
            return "STOP"

        self.current_target_subtopic = target
        target_status = self._target_status(target, state)

        # ------------------------------------------------------------
        # 1. Target is closed.
        # ------------------------------------------------------------
        # resolved = enough diagnostic evidence from the student's answer
        # non_diagnostic_response = explicitly attempted, but no usable evidence
        #
        # Both should close the target and move to the next subtopic.
        # ------------------------------------------------------------
        if target_status in {"resolved", "non_diagnostic_response"}:
            self._mark_target_done(target)
            self._reset_silence_for_target(target)

            next_target = self._next_ordered_target_subtopic()

            if next_target:
                self.current_behavior = "EXPLORE_NEW"
                self.current_target_subtopic = next_target
                self._mark_target_opened(next_target)
                return "NO"

            self.current_behavior = "OPEN"
            self.current_target_subtopic = ""
            return "STOP"

        # ------------------------------------------------------------
        # 2. Target is unseen or partially reasoned.
        # ------------------------------------------------------------
        if target_status in {"unseen", "unknown", "partially_reasoned"}:

            # If the target has not been opened yet, ask the first question
            # about that subtopic.
            if not self._target_has_been_opened(target):
                self._mark_target_opened(target)

                self.current_behavior = "EXPLORE_NEW"
                self.current_target_subtopic = target
                return "NO"

            # If the target was opened but is still unresolved, ask one
            # diagnostic follow-up.
            if self._followups_used_for_target(target) < self.max_followups_per_subtopic:
                self._increment_followup_for_target(target)

                self.current_behavior = "FOLLOW_UP"
                self.current_target_subtopic = target
                return "NO"

            # Follow-up budget exhausted. Close unresolved and move on.
            self._mark_target_done(target)
            self._reset_silence_for_target(target)

            next_target = self._next_ordered_target_subtopic()

            if next_target:
                self.current_behavior = "EXPLORE_NEW"
                self.current_target_subtopic = next_target
                self._mark_target_opened(next_target)
                return "NO"

            self.current_behavior = "OPEN"
            self.current_target_subtopic = ""
            return "STOP"

        # ------------------------------------------------------------
        # 3. Defensive fallback.
        # ------------------------------------------------------------
        self._mark_target_done(target)
        self._reset_silence_for_target(target)

        next_target = self._next_ordered_target_subtopic()

        if next_target:
            self.current_behavior = "EXPLORE_NEW"
            self.current_target_subtopic = next_target
            self._mark_target_opened(next_target)
            return "NO"

        self.current_behavior = "OPEN"
        self.current_target_subtopic = ""
        return "STOP"

    def _record_debug_state(
            self,
            state: dict,
            check: str,
            evaluated_target: str = None,
            evaluated_target_status: str = None,
    ) -> dict:
        next_target_status = self._target_status(
            self.current_target_subtopic,
            state,
        )

        debug_state = {
            "task_id": int(self.current_question.name),

            # Target evaluated from the student's latest answer
            "evaluated_target_subtopic": evaluated_target,
            "evaluated_target_status": evaluated_target_status,

            # Target after controller update
            "next_target_subtopic": self.current_target_subtopic,
            "next_target_status": next_target_status,

            # LLM diagnostic state
            "diagnostically_resolved": state.get("diagnostically_resolved", []),
            "partially_reasoned": state.get("partially_reasoned", []),
            "non_diagnostic_response": state.get("non_diagnostic_response", []),
            "unseen_subtopics": state.get("unseen_subtopics", []),
            "reason": state.get("reason", ""),

            # Controller decision
            "controller_behavior": self.current_behavior,
            "controller_decision": check,

            # Control counters
            "student_turns": self._student_turn_count_current_task(),
            "followups_used_for_target": self._followups_used_for_target(
                self.current_target_subtopic
            ),
            "silences_used_for_target": self._silences_used_for_target(
                self.current_target_subtopic
            ),
            "target_has_been_opened": self._target_has_been_opened(
                self.current_target_subtopic
            ),
            "completed_target_subtopics": list(self.completed_target_subtopics),

            # Persistent memories
            "diagnostically_resolved_memory": list(
                self.diagnostically_resolved_memory
            ),
            "partial_subtopic_memory": list(
                self.partial_subtopic_memory
            ),

        }

        self.state_history.append(debug_state)

        if self.debug_states:
            print("\n[STATE DEBUG]")
            print(json.dumps(debug_state, indent=2))

        return debug_state

    def _build_conversation_prompt(self) -> str:
        current_task = ""
        current_context = ""

        if self.current_question is not None:
            current_task = str(self.current_question.get("Task", "")).strip()
            current_context = str(self.current_question.get("Context", "")).strip()

        #current_topics = self._get_current_topics_list()
        task_window = self._get_task_window()

        return prompts["conversation_prompt"].format(
            current_task,
            current_context,
            task_window,
            self.current_target_subtopic or self._next_ordered_target_subtopic(),
            BEHAVIOR_PROMPTS.get(
                self.current_behavior,
                BEHAVIOR_PROMPTS["FOLLOW_UP"]
            ),
        )

    async def _generate_examiner_message(self) -> str:
        conversation_prompt_filled = self._build_conversation_prompt()
        user_prompt = self._get_task_window()

        if not user_prompt:
            if self.is_first_task:
                user_prompt = (
                    f"Start the oral exam naturally. "
                    f"Briefly introduce the topic '{self.current_question['Task']}' "
                    f"and ask the opening question about the current subtopic."
                )
            else:
                user_prompt = (
                    f"Naturally transition to a new topic. "
                    f"Briefly acknowledge moving on, introduce the topic "
                    f"'{self.current_question['Task']}', "
                    f"and ask the next opening question about the current subtopic."
                )
        answer_tokens = await asyncio.wait_for(
            make_call(
                system_prompt=conversation_prompt_filled,
                user_prompt=user_prompt,
                llm=MODEL,
                max_tokens=256,
            ),
            timeout=90,
        )

        answer_raw = (
            answer_tokens[0]
            if isinstance(answer_tokens, (list, tuple))
            else answer_tokens
        )

        answer = clean_string(answer_raw).strip()

        #DEBUG CODE
        #print("[SYSTEM PROMPT]", conversation_prompt_filled)
        #print("[USER PROMPT]", repr(user_prompt))
        #print("[RAW ANSWER]", answer_tokens)

        if not answer:
            answer = "Could you say that again?"

        if isinstance(answer_tokens, (list, tuple)) and len(answer_tokens) > 1:
            self.tokens += answer_tokens[1]

        self._append_turn("Examiner", answer)

        return answer

    async def send_message(self, text):
        clean_text = (text or "").strip()

        if clean_text.lower() in [
            "stop, give me feedback",
            "stop give me feedback",
            "stop feedback",
        ]:
            feedback = await self._build_feedback(max_words=300)
            return feedback + "\n[examOver]"

        # ------------------------------------------------------------
        # First examiner message for a new task.
        # ------------------------------------------------------------
        if not self.first_message:
            if not self.current_target_subtopic:
                self.current_target_subtopic = self._next_ordered_target_subtopic()

            if self.current_target_subtopic:
                self._mark_target_opened(self.current_target_subtopic)

            answer = await self._generate_examiner_message()
            self.first_message = True
            self.is_first_task = False
            return answer

        if clean_text:
            self._append_turn("Student", clean_text)
            self._reset_silence_for_target(self.current_target_subtopic)
        else:
            if (
                    self._silences_used_for_target(self.current_target_subtopic)
                    < self.max_silences_before_non_diagnostic
            ):
                self._increment_silence_for_target(self.current_target_subtopic)

                answer = "Sorry, I did not hear anything. Could you repeat that?"
                self._append_turn("Examiner", answer)
                return answer

            self._append_turn("Student", "[no response]")

        task_window = self._get_task_window()

        # ------------------------------------------------------------
        # Measure diagnostic state.
        #
        # Important:
        # Capture the target BEFORE _update_state_controls(), because
        # _update_state_controls() may immediately move current_target_subtopic
        # to the next target. This keeps debug output honest.
        # ------------------------------------------------------------
        evaluated_target = self.current_target_subtopic

        state = await self._measure_state()

        evaluated_target_status = self._target_status(
            evaluated_target,
            state,
        )

        check = self._update_state_controls(state)
        self._enforce_controller_invariants(state)

        student_turns = self._student_turn_count_current_task()

        # ------------------------------------------------------------
        # Hard cap logic.
        # ------------------------------------------------------------
        if (
                check != "STOP"
                and student_turns >= self.max_student_turns_per_task
        ):
            if self.current_target_subtopic:
                self._mark_target_done(self.current_target_subtopic)
                self._reset_silence_for_target(self.current_target_subtopic)

            self.current_behavior = "OPEN"
            self.current_target_subtopic = ""
            check = "STOP"

        self._enforce_controller_invariants(state)

        # ------------------------------------------------------------
        # Debug state.
        #
        # Requires _record_debug_state to accept:
        # evaluated_target
        # evaluated_target_status
        # ------------------------------------------------------------
        debug_state = self._record_debug_state(
            state,
            check,
            evaluated_target=evaluated_target,
            evaluated_target_status=evaluated_target_status,
        )

        self.last_debug_state = debug_state

        # ------------------------------------------------------------
        # Current rubric task finished.
        # ------------------------------------------------------------
        if check == "STOP":
            try:
                self.rubric.df.at[self.current_question.name, "Answer"] = task_window
            except Exception as e:
                print("Error saving answer:", e)

            self.silence_count_for_target = {}

            self.task_history = []
            self.first_message = False

            self.completed_target_subtopics = set()
            self.target_opened = set()

            self.diagnostically_resolved_memory = set()
            self.partial_subtopic_memory = set()
            self.target_followup_count = {}

            self.current_state = None
            self.current_behavior = "OPEN"
            self.current_target_subtopic = ""

            next_question = self.rubric.get_next_question()

            if isinstance(next_question, str) and next_question == "DONE":
                feedback = await self._build_feedback(max_words=220)
                return feedback + "\n[examOver]"

            self.current_question = next_question

            return await self.send_message("")

        # ------------------------------------------------------------
        # Continue current rubric task.
        # ------------------------------------------------------------
        examiner_message = await self._generate_examiner_message()

        # if self.debug_states:
        #     return (
        #         examiner_message
        #         + "\n\n[DEBUG]\n"
        #         + json.dumps(debug_state, indent=2)
        #     )

        return examiner_message

    async def _build_feedback(self, max_words: int = 1000) -> str:
        full_window = self._get_full_window()

        feedback_tokens = await asyncio.wait_for(
            make_call(
                system_prompt=prompts["feedback_prompt"],
                user_prompt=full_window,
                llm=MODEL,
                max_tokens=1000,
            ),
            timeout=90,
        )

        feedback_raw = (
            feedback_tokens[0]
            if isinstance(feedback_tokens, (list, tuple))
            else feedback_tokens
        )

        if isinstance(feedback_raw, str) and feedback_raw.startswith("[ERROR"):
            print(f"[DIALOG FEEDBACK ERROR] make_call returned {feedback_raw}")

        feedback = hard_cap_words(
            clean_string(feedback_raw),
            max_words=max_words,
        )

        if isinstance(feedback_tokens, (list, tuple)) and len(feedback_tokens) > 1:
            self.tokens += feedback_tokens[1]

        self.full_history.append("Examiner: " + feedback)

        return feedback




