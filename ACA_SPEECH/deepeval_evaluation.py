from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
from pydantic import BaseModel
from deepeval.models import DeepEvalBaseLLM
from mistralai.client import Mistral
import pandas as pd

from deepeval.metrics import ConversationalGEval, GEval
from deepeval.test_case import (
    ConversationalTestCase,
    Turn,
    TurnParams,
    LLMTestCase,
    LLMTestCaseParams,
)

from anthropic import Anthropic
import json

def clean_json_output(text: str) -> str:
    text = text.strip()

    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()
    elif text.startswith("```"):
        text = text.removeprefix("```").strip()

    if text.endswith("```"):
        text = text.removesuffix("```").strip()

    return text

class ClaudeJudge(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "claude-sonnet-4-5-20250929"):
        self.model_name = model_name
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def load_model(self):
        return self.client

    def generate(self, prompt: str, schema: BaseModel | None = None):
        client = self.load_model()

        if schema is not None:
            schema_json = schema.model_json_schema()

            prompt_with_schema = f"""
You must return only valid JSON that matches this Pydantic schema.

Schema:
{json.dumps(schema_json, indent=2)}

Task:
{prompt}
"""

            response = client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                temperature=0,
                messages=[
                    {"role": "user", "content": prompt_with_schema}
                ],
            )

            text = response.content[0].text
            text = clean_json_output(text)
            return schema.model_validate_json(text)

        response = client.messages.create(
            model=self.model_name,
            max_tokens=4096,
            temperature=0,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        return response.content[0].text

    async def a_generate(self, prompt: str, schema: BaseModel | None = None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.model_name

class MistralJudge(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "mistral-large-latest"):
        self.model_name = model_name
        self.client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    def load_model(self):
        return self.client

    def generate(self, prompt: str, schema: BaseModel | None = None):
        client = self.load_model()

        if schema is not None:
            response = client.chat.parse(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                response_format=schema,
                temperature=0,
            )
            return response.choices[0].message.parsed

        response = client.chat.complete(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        return response.choices[0].message.content

    async def a_generate(self, prompt: str, schema: BaseModel | None = None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.model_name

exam_metrics = [
    {
        "name": "Diagnostic questioning",
        "criteria": (
            "Evaluate whether the examiner asks sufficient and relevant questions to diagnose "
            "the student's understanding of the assessment targets. The focus is on whether the examiner gathers "
            "enough diagnostic evidence to judge whether the student has understood the concept being assessed.\n\n"

            "The assessment goals are as follows:\n\n"
            "1. Hypothesis testing:\n"
            "- Explain what the null hypothesis (H0) and alternative hypothesis (H1) represent.\n"
            "- Interpret what a p-value means in the context of hypothesis testing.\n"
            "- Explain how the significance threshold is used to draw a conclusion about H0.\n"
            "- Explain how certain we can be about the conclusion of a hypothesis test.\n\n"

            "2. Mean and median:\n"
            "- Explain how the mean is computed.\n"
            "- Explain how the median is found and how this differs for odd and even numbers.\n"
            "- Reason about how outliers affect the mean.\n"
            "- Reason about how outliers affect the median.\n\n"

            "3. Variance:\n"
            "- Explain what variance represents.\n"
            "- Explain what high and low variance represent.\n"
            "- Explain how variance and standard deviation are related.\n\n"

            "Diagnostic probing is especially valuable when:\n"
            "- The student gives a partially correct answer and it is still unclear whether they fully understand the concept.\n"
            "- The student gives a vague, short, or correct-sounding answer that may hide limited understanding.\n"
            "- The student gives an answer without explanation where explanation is needed to assess understanding.\n"
            "- The student makes a conceptual claim that should be clarified, justified, or applied more precisely.\n"
            "- The student addresses only part of the assessment target and a targeted follow-up could reveal whether they understand the missing part.\n\n"

            "Diagnostic probing is not required when:\n"
            "- The student's answer is correct and complete enough to judge the assessment target.\n"
            "- The student clearly states that they do not know the concept.\n"
            "- The student's answer provides no useful basis for further diagnostic questioning.\n"
            "- The student's answer is clearly incorrect and further probing would likely become teaching rather than assessment.\n"
            "- The examiner reasonably moves on after the student's lack of knowledge has become clear.\n"
            "- Asking another question would mainly give away the answer or lead the student toward the correct response.\n\n"

            "Observable indicators:\n"
            "- The examiner asks targeted follow-up questions when the student's understanding is not yet sufficiently clear.\n"
            "- The examiner clarifies vague, incomplete, or ambiguous answers when this could reveal understanding.\n"
            "- The examiner probes the student's conceptual understanding, not only their use of terminology.\n"
            "- The examiner checks the relevant assessment target sufficiently before moving on.\n"
            "- The examiner does not over-probe when the student has clearly shown lack of knowledge or when the answer is already complete.\n"
            "- The examiner avoids turning diagnostic probing into teaching, correction, or leading.\n\n"

            "Do not treat the examiner's decision not to correct misconceptions as poor behavior. "
            "The examiner should remain in an assessment role, not a teaching role.\n\n"

            "Score anchors:\n"
            "1.0 = The examiner consistently gathers enough diagnostic evidence for the assessment targets and asks targeted follow-ups when needed.\n"
            "0.75 = The examiner usually gathers enough diagnostic evidence, with only minor missed opportunities to probe unclear or incomplete answers.\n"
            "0.5 = The examiner gathers some diagnostic evidence, but misses several important opportunities to clarify the student's understanding.\n"
            "0.25 = The examiner often moves on before the student's understanding of the assessment target is clear.\n"
            "0.0 = The examiner does not diagnostically probe the student's understanding and repeatedly fails to gather enough evidence for assessment.\n\n"

            "In the reason, mention concrete examples where the examiner gathered enough diagnostic evidence, "
            "missed a useful opportunity to probe, or appropriately moved on because further probing was not necessary."
        ),
        "threshold": 0.7,
    },
    {
        "name": "Adaptive questioning",
        "criteria": (
            "Evaluate whether the examiner's follow-up questions are meaningfully connected"
            "to the student's previous answer within a statistical task or topic. Do not measure whether follow-up questions are generally asked but if they are asked, whether they link to the previous answer."
            "Important is that neutrality is important here, so neutral behavior from the examiner is correct. \n\n"
            "Follow-up questions should be evaluated in relation to the following topics:\n\n"
            "1. Hypothesis testing:\n"
            "- Explain what the null hypothesis (H0) and alternative hypothesis (H1) represent.\n"
            "- Interpret what a p-value means in the context of hypothesis testing.\n"
            "- Explain how the significance threshold is used to draw a conclusion about H0.\n"
            "- Explain how certain we can be about the conclusion of a hypothesis test.\n\n"
            "2. Mean and median:\n"
            "- Explain how the mean is computed.\n"
            "- Explain how the median is found and how this differs for odd and even numbers.\n"
            "- Reason about how outliers affect the mean.\n"
            "- Reason about how outliers affect the median.\n\n"
           "3. Variance:\n"
            "- Explain what variance represents.\n"
            "- Explain what high and low variance represent.\n"
            "- Explain how variance and standard deviation are related.\n\n"
            "Observable indicators:\n"
            "- Follow-up questions respond to what the student actually said.\n"
            "- The examiner is specific about the missing part of the student's answer.\n"
            "- The examiner avoids generic follow-ups that could be asked regardless of the student's answer.\n"
            "- The examiner avoids unnecessary repetition unless it serves a clear diagnostic purpose.\n\n"
            "Do not treat the examiner's decision not to correct misconceptions as poor behavior. During the oral exam, the examiner should remain in an assessment role."
            "Score anchors:\n"
            "1.0 = Follow-up questions are consistently relevant and adaptive to the student's answers.\n"
            "0.75 = Follow-up questions are mostly adaptive, with minor generic or missed moments.\n"
            "0.5 = Some follow-ups are adaptive, but others are generic, repetitive, or insufficiently connected.\n"
            "0.25 = Follow-ups are mostly generic or poorly connected to the student's answer.\n"
            "0.0 = The examiner does not adapt to the student's answers.\n\n"
            "In the reason, explain how the follow-up questions did or did not build on the student's previous answers."
        ),
        "threshold": 0.7,
    },
    {
        "name": "Neutrality and non-leading prompting",
        "criteria": (
            "Evaluate whether the examiner probes the student's understanding without giving away "
            "key information about the expected answer or influencing the student's response.\n\n"
            "Observable indicators:\n"
            "- The examiner does not reveal the correct answer before the student has shown it.\n"
            "- The examiner does not give hints about the answer.\n"
            "- The examiner avoids leading questions that point to the expected answer.\n"
            "- The examiner remains neutral and avoids excessive praise, reassurance, discouragement, or evaluative reactions during the exam.\n"
            "Score anchors:\n"
            "1.0 = The examiner remains neutral and non-leading throughout the interaction.\n"
            "0.75 = The examiner is mostly neutral, with only minor leading or overly reassuring moments.\n"
            "0.5 = The examiner sometimes remains neutral, but also gives noticeable hints, praise, or guidance.\n"
            "0.25 = The examiner often leads the student or gives away important parts of the answer.\n"
            "0.0 = The examiner repeatedly supplies answers or strongly directs the student toward the expected response.\n\n"
            "In the reason, identify any hints, answer leakage, praise, reassurance, or discouraging reactions that affected the score."
        ),
        "threshold": 0.7,
    },
    {
        "name": "Content alignment",
        "criteria": (
            "Evaluate whether the examiner keeps the oral exam aligned with the intended statistical content and "
            "assessment goals. The assessment goals are:\n\n"
            "1. Hypothesis testing:\n"
            "- Explain what the null hypothesis (H0) and alternative hypothesis (H1) represent.\n"
            "- Interpret what a p-value means in the context of hypothesis testing.\n"
            "- Explain how the significance threshold is used to draw a conclusion about H0.\n"
            "- Explain how certain we can be about the conclusion of a hypothesis test.\n\n"
            "2. Mean and median:\n"
            "- Explain how the mean is computed.\n"
            "- Explain how the median is found and how this differs for odd and even numbers.\n"
            "- Reason about how outliers affect the mean.\n"
            "- Reason about how outliers affect the median.\n\n"
            "3. Variance:\n"
            "- Explain what variance represents.\n"
            "- Explain what high and low variance represent.\n"
            "- Explain how variance and standard deviation are related.\n\n"
            "Observable indicators:\n"
            "- Questions focus on relevant statistical concepts, which are hypothesis testing, mean and median, and variance.\n"
            "- The examiner avoids unrelated topics or overly minor details outside the intended scope.\n"
            "- The examiner's questions match the current task and learning goals.\n"
            "- The examiner does not drift away from the construct being assessed.\n\n"
            "Score anchors:\n"
            "1.0 = The interaction is fully aligned with the intended content and learning goals.\n"
            "0.75 = The interaction is mostly aligned, with only minor deviations.\n"
            "0.5 = The interaction is partly aligned, but includes some irrelevant, overly minor, or off-scope content.\n"
            "0.25 = The interaction often drifts away from the intended content.\n"
            "0.0 = The interaction is largely unrelated to the intended assessment content.\n\n"
            "In the reason, explain whether the questions stayed aligned with the rubric and reference document."
        ),
        "threshold": 0.7,
    },
    {
        "name": "Assessment coherence",
        "criteria": (
            "Evaluate whether the oral exam is conducted in a coherent and understandable sequence.\n\n"
            "Observable indicators:\n"
            "- Questions within each statistical task follow a logical order and are coherent.\n"
            "- Topic transitions are understandable to the student, by telling the student that they are moving to a new topic.\n"
            "- The examiner does not jump abruptly between unrelated statistical concepts. Abrupt means not announcing a change.\n"
            "- The examiner responds to what the student says.\n\n"
            "Score anchors:\n"
            "1.0 = The exam is consistently coherent, with logical sequencing and clear transitions.\n"
            "0.75 = The exam is mostly coherent, with only minor unclear transitions.\n"
            "0.5 = The exam is partly coherent, but some sequencing or transitions are confusing.\n"
            "0.25 = The exam often feels disjointed or difficult to follow.\n"
            "0.0 = The exam is incoherent or confusing as a whole.\n\n"
            "In the reason, explain how the sequence of questions and transitions affected the score. One example can only affect the score once, so if there are multiple examples, explain how they together influenced the score."
        ),
        "threshold": 0.7,
    },
    {
        "name": "Feedback quality",
        "criteria": (
            "Evaluate whether the final feedback is accurate, specific, clear, balanced, and aligned "
            "with the student's actual performance in the conversation.\n\n"
            "Observable indicators:\n"
            "- Feedback accurately reflects what the student did and did not demonstrate.\n"
            "- Feedback identifies relevant strengths and weaknesses.\n"
            "- Feedback distinguishes correct, incomplete, and unclear understanding.\n"
            "- Feedback gives useful and specific information for improvement.\n"
            "- Feedback is not generic or disconnected from the conversation.\n\n"
            "Score anchors:\n"
            "1.0 = Feedback is accurate, specific, balanced, clear, and useful for improvement.\n"
            "0.75 = Feedback is mostly high quality, with minor missing detail or specificity.\n"
            "0.5 = Feedback is partly useful but somewhat generic, incomplete, or only loosely connected to the conversation.\n"
            "0.25 = Feedback is mostly generic, vague, inaccurate, or poorly aligned.\n"
            "0.0 = Feedback is absent, misleading, or unrelated to the student's actual performance.\n\n"
            "In the reason, explain which parts of the feedback were accurate, specific, balanced, and useful."
        ),
        "threshold": 0.7,
    },
]

role_map = {
    "bot": "assistant",
    "user": "user",
    "feedback": "feedback",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one CSV conversation or a folder of CSV conversations with DeepEval."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to one input CSV file or a folder containing CSV conversations.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save results.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="If --input is a folder, also evaluate CSV files in subfolders.",
    )

    parser.add_argument(
        "--metric-config",
        default=None,
        help="Optional JSON file with metric definitions (name, criteria, threshold).",
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Optional evaluation model name passed to DeepEval.",
    )

    parser.add_argument(
        "--chatbot-role",
        default="An oral exam examiner who assesses a student's knowledge through probing questions, "
        "guides without revealing answers, and follows the intended exam structure.",
        help="Description of the bot's role.",
    )

    parser.add_argument(
        "--user-description",
        default="A student answering oral exam questions.",
        help="Description of the user.",
    )

    parser.add_argument(
        "--strict-mode",
        action="store_true",
        help="Use binary scoring instead of continuous scoring.",
    )

    parser.add_argument(
        "--verbose-mode",
        action="store_true",
        help="Show DeepEval judge steps.",
    )

    parser.add_argument(
        "--rubric",
        default=None,
        help="Optional path to a rubric text file.",
    )

    parser.add_argument(
        "--document",
        default=None,
        help="Optional path to a reference document text file.",
    )

    return parser.parse_args()

def load_metrics(metric_config: Optional[str]) -> List[Dict[str, Any]]:
    if metric_config is None:
        return exam_metrics

    with open(metric_config, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError("Metric config must be a non-empty JSON list.")

    metrics: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict) or "name" not in item or "criteria" not in item:
            raise ValueError("Each metric must be an object with 'name' and 'criteria'.")
        metrics.append(
            {
                "name": str(item["name"]),
                "criteria": str(item["criteria"]),
                "threshold": float(item.get("threshold", 0.5)),
            }
        )
    return metrics


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def load_support_text(path: Optional[str]) -> str:
    if path is None:
        return ""
    return Path(path).read_text(encoding="utf-8").strip()


def load_conversation(csv_path: str) -> List[Dict[str, str]]:
    df = pd.read_csv(csv_path)

    required_columns = {"role", "content"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV must contain the columns {sorted(required_columns)}. Missing: {sorted(missing)}"
        )

    rows: List[Dict[str, str]] = []

    for i, row in df.iterrows():
        raw_role = clean_text(row["role"]).lower()
        content = clean_text(row["content"])

        if not raw_role or not content:
            continue

        if raw_role not in {"bot", "user", "feedback"}:
            raise ValueError(
                f"Invalid role '{raw_role}' on CSV row {i + 2}. "
                "Allowed roles are: ['bot', 'user', 'feedback']"
            )

        rows.append(
            {
                "role": raw_role,
                "content": content,
            }
        )

    if len(rows) < 2:
        raise ValueError("Need at least 2 non-empty rows to evaluate a conversation.")

    return rows

def build_metric(metric, model, strict_mode, verbose_mode, rubric_text="", document_text=""):
    criteria = add_reasoning_instructions(metric["criteria"])

    if metric["name"] in {"Content alignment"}:
        extra_context_parts = []
        if rubric_text:
            extra_context_parts.append(f"Rubric:\n{rubric_text}")
        if document_text:
            extra_context_parts.append(f"Reference document:\n{document_text}")
        if extra_context_parts:
            criteria += (
                "\n\nUse the following materials as the knowledge base for this evaluation:\n\n"
                + "\n\n".join(extra_context_parts)
            )

    kwargs = {
        "name": metric["name"],
        "criteria": criteria,
        "threshold": float(metric.get("threshold", 0.5)),
        "evaluation_params": [TurnParams.CONTENT],
        "strict_mode": strict_mode,
        "verbose_mode": verbose_mode,
    }
    if model:
        kwargs["model"] = model
    return ConversationalGEval(**kwargs)

def add_reasoning_instructions(criteria: str) -> str:
    return f"""
You are an evaluation judge. You must follow these rules strictly.

GLOBAL SCORING RULES:
1. Evaluate only the examiner's behaviour.
2. Do not lower the score because the student gives incorrect, incomplete, vague, or uncertain answers.
3. Student answers may only be used as evidence for whether the examiner responded appropriately.
4. The score must reflect what the examiner did, not how well the student performed.
5. Do not give a high score only because the examiner sounded fluent.
6. Use the score anchors in the metric. Do not invent a different scoring standard.
7. If there is no evidence for a weakness, do not invent one.

REQUIRED REASON STRUCTURE:
1. Score interpretation: explain what the numerical score means.
2. Evidence: cite concrete examiner behaviour from the conversation.
3. Weaknesses: explain what lowered the score, if anything.
4. Final justification: explain why this score fits the score anchor and why a higher or lower score was not chosen.

METRIC-SPECIFIC CRITERIA:
{criteria}
"""

def build_feedback_metric(metric, model, strict_mode, verbose_mode, rubric_text="", document_text=""):
    criteria = add_reasoning_instructions(metric["criteria"])

    if rubric_text or document_text:
        extra_context_parts = []
        if rubric_text:
            extra_context_parts.append(f"Rubric:\n{rubric_text}")
        if document_text:
            extra_context_parts.append(f"Reference document:\n{document_text}")
        criteria += (
            "\n\nUse the following materials as additional context if relevant:\n\n"
            + "\n\n".join(extra_context_parts)
        )

    kwargs = {
        "name": metric["name"],
        "criteria": criteria,
        "threshold": float(metric.get("threshold", 0.5)),
        "evaluation_params": [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        "strict_mode": strict_mode,
        "verbose_mode": verbose_mode,
    }
    if model:
        kwargs["model"] = model
    return GEval(**kwargs)


def run_evaluation(turns, metrics, model, chatbot_role, user_description, strict_mode, verbose_mode,
                   rubric_text="", document_text=""):
    conversation_turns = [
        Turn(
            role=role_map[turn["role"]],
            content=turn["content"],
        )
        for turn in turns
        if turn["role"] != "feedback"
    ]

    feedback_turns = [
        turn for turn in turns
        if turn["role"] == "feedback"
    ]

    test_case = ConversationalTestCase(
        turns=conversation_turns,
        chatbot_role=chatbot_role,
        user_description=user_description,
    )

    rows = []
    for metric_spec in metrics:
        row = {
            "metric_name": metric_spec["name"],
            "threshold": float(metric_spec.get("threshold", 0.5)),
            "n_turns": len(turns),
            "score": None,
            "score_label": None,
            "success": None,
            "reason": None,
            "error": None,
        }
        try:
            if metric_spec["name"] == "Feedback quality":
                if not feedback_turns:
                    raise ValueError("No feedback turn found, but Feedback quality metric was requested.")

                last_feedback = feedback_turns[-1]
                last_feedback_index = turns.index(last_feedback)

                previous_turns = [
                    turn for turn in turns[:last_feedback_index]
                    if turn["role"] != "feedback"
                ]

                conversation_text = "\n".join(
                    f"{turn['role']}: {turn['content']}" for turn in previous_turns
                )

                metric = build_feedback_metric(
                    metric_spec,
                    model,
                    strict_mode,
                    verbose_mode,
                    rubric_text,
                    document_text,
                )
                feedback_test_case = LLMTestCase(
                    input=conversation_text,
                    actual_output=last_feedback["content"],
                )
                metric.measure(feedback_test_case)
            else:
                metric = build_metric(
                    metric_spec,
                    model,
                    strict_mode,
                    verbose_mode,
                    rubric_text,
                    document_text,
                )
                metric.measure(test_case)

            row["score"] = getattr(metric, "score", None)
            row["success"] = getattr(metric, "success", None)
            row["reason"] = getattr(metric, "reason", None)

            score = row["score"]
            if score is not None:
                if score >= 0.85:
                    row["score_label"] = "strong"
                elif score >= 0.70:
                    row["score_label"] = "sufficient/good"
                elif score >= 0.50:
                    row["score_label"] = "mixed/partially sufficient"
                elif score >= 0.25:
                    row["score_label"] = "weak"
                else:
                    row["score_label"] = "very weak"
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    return pd.DataFrame(rows)


def save_outputs(output_dir, results_df, turns, config):
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / "conversation_metric_results.csv", index=False)

    summary = pd.DataFrame(
        {
            "n_metrics": [len(results_df)],
            "n_turns": [len(turns)],
            "mean_score": [pd.to_numeric(results_df["score"], errors="coerce").mean()],
            "n_errors": [results_df["error"].notna().sum()],
        }
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    with open(output_dir / "parsed_conversation.json", "w", encoding="utf-8") as f:
        json.dump(
            turns,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def evaluate_one_file(csv_path: Path, base_output_dir: Path, args) -> Dict[str, Any]:
    turns = load_conversation(str(csv_path))
    metrics = load_metrics(args.metric_config)
    rubric_text = load_support_text(args.rubric)
    document_text = load_support_text(args.document)

    judge_model = args.model

    if args.model and args.model.startswith("mistral"):
        judge_model = MistralJudge(args.model)
    elif args.model and args.model.startswith("claude"):
        judge_model = ClaudeJudge(args.model)
    elif args.model and args.model.startswith("openrouter/"):
        raise ValueError("OpenRouterJudge is not defined in this script yet.")

    results_df = run_evaluation(
        turns=turns,
        metrics=metrics,
        model=judge_model,
        chatbot_role=args.chatbot_role,
        user_description=args.user_description,
        strict_mode=args.strict_mode,
        verbose_mode=args.verbose_mode,
        rubric_text=rubric_text,
        document_text=document_text,
    )

    conversation_name = csv_path.stem
    output_dir = base_output_dir / conversation_name

    config = {
        "input": str(csv_path.resolve()),
        "model": args.model,
        "chatbot_role": args.chatbot_role,
        "user_description": args.user_description,
        "strict_mode": args.strict_mode,
        "verbose_mode": args.verbose_mode,
        "rubric": str(Path(args.rubric).resolve()) if args.rubric else None,
        "document": str(Path(args.document).resolve()) if args.document else None,
        "metrics": metrics,
    }

    save_outputs(output_dir, results_df, turns, config)

    results_df.insert(0, "conversation_file", csv_path.name)
    results_df.insert(1, "conversation_path", str(csv_path.resolve()))

    return {
        "conversation_file": csv_path.name,
        "conversation_path": str(csv_path.resolve()),
        "n_metrics": len(results_df),
        "n_turns": len(turns),
        "mean_score": pd.to_numeric(results_df["score"], errors="coerce").mean(),
        "n_errors": results_df["error"].notna().sum(),
        "results": results_df,
    }

def main():
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        if input_path.suffix.lower() != ".csv":
            raise ValueError("--input must be a CSV file or a folder containing CSV files.")

        result = evaluate_one_file(input_path, output_dir, args)

        print(f"Saved results for: {input_path.name}")
        print(f"Saved output folder to: {output_dir / input_path.stem}")
        return 0

    if input_path.is_dir():
        pattern = "**/*.csv" if args.recursive else "*.csv"
        csv_files = sorted(input_path.glob(pattern))

        if not csv_files:
            raise ValueError(f"No CSV files found in folder: {input_path}")

        all_results = []
        batch_rows = []

        for csv_path in csv_files:
            print(f"Evaluating: {csv_path}")

            try:
                result = evaluate_one_file(csv_path, output_dir, args)

                all_results.append(result["results"])

                batch_rows.append(
                    {
                        "conversation_file": result["conversation_file"],
                        "conversation_path": result["conversation_path"],
                        "n_metrics": result["n_metrics"],
                        "n_turns": result["n_turns"],
                        "mean_score": result["mean_score"],
                        "n_errors": result["n_errors"],
                        "status": "success",
                        "error": None,
                    }
                )

            except Exception:
                batch_rows.append(
                    {
                        "conversation_file": csv_path.name,
                        "conversation_path": str(csv_path.resolve()),
                        "n_metrics": None,
                        "n_turns": None,
                        "mean_score": None,
                        "n_errors": None,
                        "status": "error",
                        "error": traceback.format_exc(),
                    }
                )

        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
            combined_results.to_csv(
                output_dir / "all_conversation_metric_results.csv",
                index=False,
                encoding="utf-8",
            )

        batch_summary = pd.DataFrame(batch_rows)
        batch_summary.to_csv(
            output_dir / "batch_summary.csv",
            index=False,
            encoding="utf-8",
        )

        print(f"Evaluated {len(csv_files)} CSV file(s).")
        print(f"Saved batch summary to: {output_dir / 'batch_summary.csv'}")
        print(f"Saved combined results to: {output_dir / 'all_conversation_metric_results.csv'}")
        return 0

    raise ValueError(f"Input path does not exist: {input_path}")

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("Interrupted by user.")
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)