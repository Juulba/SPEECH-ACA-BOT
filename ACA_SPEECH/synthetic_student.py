import asyncio
from typing import Optional
import random

from setup.constants import MODEL
from LLM.llm_calls import make_call
from utils import clean_string
from knowledge_state import SUBTOPIC_KNOWLEDGE


class SyntheticStudent:
    def __init__(self, profile: Optional[dict] = None):
        self.profile = profile or {
            "knowledge_state_probs": {
                "correct": 0.35,
                "partial": 0.40,
                "wrong": 0.15,
                "no_knowledge": 0.10,
            },
        }

        self.tokens = 0
        self.sampled_subtopic_states = {}
        self.subtopic_turn_counts = {}

        self.sample_all_subtopics()

    def sample_all_subtopics(self) -> dict:
        """
        Sample one fixed knowledge state for every known subtopic.

        This creates the synthetic student's full latent knowledge profile
        before the exam starts.
        """
        for subtopic in SUBTOPIC_KNOWLEDGE:
            if subtopic not in self.sampled_subtopic_states:
                self.sampled_subtopic_states[subtopic] = self._sample_state()

        return dict(self.sampled_subtopic_states)


    def _sample_state(self) -> str:
        probs = self.profile.get("knowledge_state_probs", {})
        states = list(probs.keys())
        weights = list(probs.values())
        return random.choices(states, weights=weights, k=1)[0]

    def _get_subtopic_knowledge(self, current_subtopic: str) -> str:
        if not current_subtopic:
            return "You have no specific remembered knowledge for this question."

        if current_subtopic not in self.sampled_subtopic_states:
            self.sampled_subtopic_states[current_subtopic] = self._sample_state()

        state = self.sampled_subtopic_states[current_subtopic]

        if current_subtopic not in SUBTOPIC_KNOWLEDGE:
            print(f"[WARNING] No sampled knowledge found for subtopic: {current_subtopic}")
            return "You have no specific remembered knowledge for this question."

        return SUBTOPIC_KNOWLEDGE[current_subtopic].get(
            state,
            "You have no specific remembered knowledge for this question."
        )

    def get_sampled_knowledge_state(self, current_subtopic: str = "") -> dict:
        """
        Return the sampled synthetic knowledge state.

        Used only for simulation logging.
        """
        current_subtopic = (current_subtopic or "").strip()

        return {
            "current_subtopic": current_subtopic or None,
            "current_state": (
                self.sampled_subtopic_states.get(current_subtopic)
                if current_subtopic
                else None
            ),
            "states_by_subtopic": dict(self.sampled_subtopic_states),
        }

    def _build_system_prompt(self):
        response_style = self.profile.get("response_style", "stepwise_incomplete")

        return f"""
        You are simulating a student in an oral exam.

        General rules:
        - Answer only as the student.
        - Answer in a natural spoken exam style.

        Stepwise answering rule:
        - Give answers in small steps rather than all at once.
        - On the first question about a subtopic, answer only part of the answer.
        - Leave some reasoning implicit unless the examiner asks a follow-up question.
        - If the examiner asks a follow-up question, add one additional reasoning step.
        - Do not suddenly give a perfect full answer unless your sampled knowledge is correct and the examiner has explicitly asked for explanation or reasoning.
        - If your sampled knowledge is partial, keep the answer incomplete even after follow-up.
        - If your sampled knowledge is wrong, answer according to that wrong belief.
        - If your sampled knowledge is no_knowledge, say you do not know or give a very vague answer.

        Consistency rule:
        - During a follow-up question, do not change your sampled knowledge state for the subtopic.
        - You may elaborate slightly, but only within the limits of the same sampled knowledge.
        """.strip()

    def _build_user_prompt(
            self,
            examiner_message: str,
            current_task: str,
            current_subtopic: str,
            current_context: str = "",
            subtopic_turn_count: int = 1,
    ) -> str:

        sampled_knowledge = self._get_subtopic_knowledge(current_subtopic)

        return f"""
    Current task:
    {current_task}

    Current subtopic:
    {current_subtopic}

    This is your answer number {subtopic_turn_count} for this subtopic.

    Recent conversation context:
    {current_context}

    This is what you currently remember about this subtopic:
    {sampled_knowledge}

    Examiner question:
    {examiner_message}

    Answer as student a student would in an oral exam, in a partial and concise way.

    Important:
    - Answer only the current subtopic.
    - Do not answer other subtopics, even if they are related.
    - Do not combine multiple rubric topics in one answer.
    - The sampled knowledge is divided into steps.
    - If this is answer number 1 for the subtopic, use only Step 1.
    - If this is answer number 2, add only Step 2.
    - If this is answer number 3, add only Step 3.
    - If this is answer number 4, add only Step 4.
    - Never combine multiple steps in one answer.
    - If there is no next step available, say that you are not sure what else to add.
    """.strip()

    def _postprocess_answer(self, answer: str) -> str:
        answer = clean_string(answer).strip()

        if not answer:
            return "I'm not sure."

        return answer

    async def answer(
            self,
            examiner_message: str,
            current_task: str = "",
            current_subtopic: str = "",
            current_context: str = "",
    ) -> str:
        current_subtopic_key = (current_subtopic or "").strip()

        if current_subtopic_key:
            self.subtopic_turn_counts[current_subtopic_key] = (
                    self.subtopic_turn_counts.get(current_subtopic_key, 0) + 1
            )

        subtopic_turn_count = self.subtopic_turn_counts.get(current_subtopic_key, 1)

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            examiner_message=examiner_message,
            current_task=current_task,
            current_subtopic=current_subtopic,
            current_context=current_context,
            subtopic_turn_count=subtopic_turn_count,
        )

        answer_tokens = await asyncio.wait_for(
            make_call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                llm=MODEL,
            ),
            timeout=90,
        )

        raw_answer = answer_tokens[0]
        self.tokens += answer_tokens[1]

        return self._postprocess_answer(raw_answer)
