from __future__ import annotations

import argparse
import itertools
import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel

from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from mistralai.client import Mistral


# ---------------------------------------------------------------------
# Define your groups here
# ---------------------------------------------------------------------

CONVERSATION_GROUPS = {
    "correct": [
        # "exports/llm_student_correct_0.csv",
        # "exports/llm_student_correct_1.csv",
    ],
    "partial": [
        # "exports/llm_student_partial_0.csv",
        # "exports/llm_student_partial_1.csv",
    ],
    "wrong": [
        # "exports/llm_student_wrong_0.csv",
        # "exports/llm_student_wrong_1.csv",
    ],
    "no_knowledge": [
        # "exports/llm_student_no_knowledge_0.csv",
        # "exports/llm_student_no_knowledge_1.csv",
    ],
}


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
                messages=[{"role": "user", "content": prompt}],
                response_format=schema,
                temperature=0.2,
            )
            return response.choices[0].message.parsed

        response = client.chat.complete(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
        )

        return response.choices[0].message.content

    async def a_generate(self, prompt: str, schema: BaseModel | None = None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.model_name


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def load_conversation(csv_path: str) -> List[Dict[str, str]]:
    df = pd.read_csv(csv_path)

    required_columns = {"role", "content"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV must contain the columns {sorted(required_columns)}. "
            f"Missing: {sorted(missing)}"
        )

    rows = []

    for i, row in df.iterrows():
        role = clean_text(row["role"]).lower()
        content = clean_text(row["content"])

        if not role or not content:
            continue

        if role not in {"bot", "user", "feedback"}:
            raise ValueError(
                f"Invalid role '{role}' on CSV row {i + 2}. "
                "Allowed roles are: bot, user, feedback."
            )

        # Feedback is not needed for prompting consistency.
        if role == "feedback":
            continue

        rows.append(
            {
                "role": role,
                "content": content,
            }
        )

    if len(rows) < 2:
        raise ValueError(f"Need at least 2 non-empty turns in {csv_path}")

    return rows


def conversation_to_text(turns: List[Dict[str, str]]) -> str:
    role_names = {
        "bot": "examiner",
        "user": "student",
    }

    return "\n".join(
        f"{role_names.get(turn['role'], turn['role'])}: {turn['content']}"
        for turn in turns
    )


def score_label(score: Optional[float]) -> Optional[str]:
    if score is None:
        return None
    if score >= 0.85:
        return "strong"
    if score >= 0.70:
        return "sufficient/good"
    if score >= 0.50:
        return "mixed/partially sufficient"
    if score >= 0.25:
        return "weak"
    return "very weak"


def build_judge_model(model_name: Optional[str]):
    if model_name and model_name.startswith("mistral"):
        return MistralJudge(model_name)
    return model_name


def build_prompting_consistency_metric(model, strict_mode=False, verbose_mode=False):
    criteria = """
Evaluate whether two synthetic students with the same knowledge background received comparable prompting opportunities from the examiner.

The goal is not to evaluate whether the students performed well. The goal is to evaluate whether the examiner behaved consistently towards comparable levels of demonstrated understanding.

Observable indicators:
- Similar incomplete answers receive similar opportunities for clarification or elaboration.
- Similar correct answers receive similar closure or follow-up.
- Similar vague answers receive similar probing.
- The examiner does not give one student substantially more help, hints, scaffolding, or follow-up than the other without a clear reason.
- Differences in prompting are acceptable when they are justified by differences in what the student actually demonstrated.

Important scoring rule:
Only evaluate examiner behaviour. Do not lower the score because one student gives incorrect, incomplete, vague, or uncertain answers. Student answers may only be used as evidence for whether the examiner responded consistently and appropriately.

Score anchors:
1.0 = The examiner treats comparable student answers consistently across both conversations.
0.75 = The examiner is mostly consistent, with only minor unexplained differences.
0.5 = The examiner is partly consistent, but there are several differences in prompting opportunities.
0.25 = The examiner is often inconsistent across comparable student answers.
0.0 = The examiner's prompting is clearly arbitrary or unfair across the two conversations.

In the reason, compare concrete moments from both conversations. Explain whether any differences in examiner prompting were justified by differences in student answers.
"""

    kwargs = {
        "name": "Prompting consistency",
        "criteria": criteria,
        "threshold": 0.7,
        "evaluation_params": [LLMTestCaseParams.INPUT],
        "strict_mode": strict_mode,
        "verbose_mode": verbose_mode,
    }

    if model:
        kwargs["model"] = model

    return GEval(**kwargs)


def make_pairs_from_groups(groups: Dict[str, List[str]]):
    pairs = []

    for knowledge_level, file_paths in groups.items():
        if len(file_paths) < 2:
            print(
                f"Skipping group '{knowledge_level}' because it contains fewer than 2 conversations."
            )
            continue

        for file_a, file_b in itertools.combinations(file_paths, 2):
            pairs.append(
                {
                    "knowledge_level": knowledge_level,
                    "file_path_a": file_a,
                    "file_path_b": file_b,
                }
            )

    return pairs


def run_pairwise_evaluation(
    groups: Dict[str, List[str]],
    output_dir: str,
    model_name: Optional[str],
    strict_mode: bool,
    verbose_mode: bool,
):
    judge_model = build_judge_model(model_name)
    pairs = make_pairs_from_groups(groups)

    rows = []

    for pair in pairs:
        file_a = pair["file_path_a"]
        file_b = pair["file_path_b"]
        knowledge_level = pair["knowledge_level"]

        row = {
            "metric_name": "Prompting consistency",
            "model": model_name,
            "knowledge_level": knowledge_level,
            "conversation_a": Path(file_a).stem,
            "conversation_b": Path(file_b).stem,
            "file_path_a": file_a,
            "file_path_b": file_b,
            "score": None,
            "score_label": None,
            "success": None,
            "reason": None,
            "error": None,
        }

        try:
            turns_a = load_conversation(file_a)
            turns_b = load_conversation(file_b)

            text_a = conversation_to_text(turns_a)
            text_b = conversation_to_text(turns_b)

            pair_input = f"""
The following two conversations are from synthetic students with the same knowledge background: {knowledge_level}.

Compare whether the examiner gave comparable prompting opportunities in both conversations.

Conversation A:
{text_a}

Conversation B:
{text_b}
"""

            metric = build_prompting_consistency_metric(
                model=judge_model,
                strict_mode=strict_mode,
                verbose_mode=verbose_mode,
            )

            test_case = LLMTestCase(input=pair_input)
            metric.measure(test_case)

            row["score"] = getattr(metric, "score", None)
            row["success"] = getattr(metric, "success", None)
            row["reason"] = getattr(metric, "reason", None)
            row["score_label"] = score_label(row["score"])

        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"

        rows.append(row)

    results_df = pd.DataFrame(rows)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        output_path / "prompting_consistency_pair_results.csv",
        index=False,
    )

    summary_df = pd.DataFrame(
        {
            "model": [model_name],
            "n_pairs": [len(results_df)],
            "mean_score": [
                pd.to_numeric(results_df["score"], errors="coerce").mean()
            ],
            "n_errors": [results_df["error"].notna().sum()],
        }
    )

    summary_df.to_csv(
        output_path / "prompting_consistency_summary.csv",
        index=False,
    )

    with open(
        output_path / "prompting_consistency_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "model": model_name,
                "strict_mode": strict_mode,
                "verbose_mode": verbose_mode,
                "conversation_groups": groups,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved pair results to: {output_path / 'prompting_consistency_pair_results.csv'}")
    print(f"Saved summary to: {output_path / 'prompting_consistency_summary.csv'}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate prompting consistency by comparing pairs of synthetic student conversations."
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save pairwise consistency results.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Judge model name.",
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

    return parser.parse_args()


def main():
    args = parse_args()

    run_pairwise_evaluation(
        groups=CONVERSATION_GROUPS,
        output_dir=args.output_dir,
        model_name=args.model,
        strict_mode=args.strict_mode,
        verbose_mode=args.verbose_mode,
    )

    return 0


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
