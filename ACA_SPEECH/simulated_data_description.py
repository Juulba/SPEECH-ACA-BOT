#!/usr/bin/env python3
"""
Compute descriptive conversation metrics for a folder of conversation CSV files.

For each CSV file, the script calculates:
- number of turns
- user response length in words
- examiner response length in words
- feedback length in words

It then writes:
1. per_conversation_descriptives.csv
2. descriptive_summary.csv

Example:
python conversation_descriptives.py \
  --input "/Users/JulieB/PycharmProjects/ACA_SPEECH/exports_simulation_final_version-3-9" \
  --output-dir "/Users/JulieB/Desktop/Thesis Data/descriptive_results"
"""

import argparse
import re
from pathlib import Path

import pandas as pd


def count_words(text) -> int:
    """Count words in a robust but simple way."""
    if pd.isna(text):
        return 0

    text = str(text).strip()

    if not text:
        return 0

    return len(re.findall(r"\b\w+(?:['-]\w+)?\b", text))


def find_content_column(df: pd.DataFrame) -> str:
    """Find the column that contains the utterance text."""
    possible = ["content", "text", "message", "utterance", "transcript"]

    for col in possible:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not find a text/content column. Columns found: {list(df.columns)}"
    )


def normalise_role(role) -> str:
    """Standardise role names across slightly different export formats."""
    role = str(role).strip().lower()

    if role in {"user", "student", "participant"}:
        return "user"

    if role in {"bot", "assistant", "examiner", "aca-bot", "aca_bot"}:
        return "examiner"

    if role in {"feedback", "final_feedback"}:
        return "feedback"

    return role


def process_file(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)

    if "role" not in df.columns:
        raise ValueError(f"{csv_path.name}: missing required column 'role'.")

    content_col = find_content_column(df)

    df["role_norm"] = df["role"].apply(normalise_role)
    df["word_count"] = df[content_col].apply(count_words)

    user_rows = df[df["role_norm"] == "user"]
    examiner_rows = df[df["role_norm"] == "examiner"]
    feedback_rows = df[df["role_norm"] == "feedback"]

    # Turns = user + examiner utterances.
    # Feedback is calculated separately and not counted as a dialogue turn.
    n_turns = len(user_rows) + len(examiner_rows)

    return {
        "conversation_file": csv_path.name,
        "n_turns": n_turns,
        "n_user_turns": len(user_rows),
        "n_examiner_turns": len(examiner_rows),
        "user_response_length_mean_words": user_rows["word_count"].mean(),
        "user_response_length_sd_words": user_rows["word_count"].std(ddof=1),
        "examiner_response_length_mean_words": examiner_rows["word_count"].mean(),
        "examiner_response_length_sd_words": examiner_rows["word_count"].std(ddof=1),
        "feedback_length_words": feedback_rows["word_count"].sum(),
        "has_feedback": len(feedback_rows) > 0,
    }


def summarise(per_conversation: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []

    metrics = [
        "n_turns",
        "n_user_turns",
        "n_examiner_turns",
        "user_response_length_mean_words",
        "examiner_response_length_mean_words",
        "feedback_length_words",
    ]

    for metric in metrics:
        values = per_conversation[metric].dropna()

        summary_rows.append(
            {
                "metric": metric,
                "n_conversations": len(values),
                "mean": values.mean(),
                "sd": values.std(ddof=1),
                "min": values.min(),
                "max": values.max(),
            }
        )

    return pd.DataFrame(summary_rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Folder containing conversation CSV files.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Folder where output CSV files should be saved.",
    )

    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="File pattern to include. Default: *.csv",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob(args.pattern))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {input_dir} with pattern {args.pattern}"
        )

    rows = []
    skipped = []

    for csv_path in csv_files:
        try:
            rows.append(process_file(csv_path))
        except Exception as e:
            skipped.append(
                {
                    "conversation_file": csv_path.name,
                    "error": str(e),
                }
            )

    per_conversation = pd.DataFrame(rows)

    if per_conversation.empty:
        raise RuntimeError("No files could be processed successfully.")

    summary = summarise(per_conversation)

    per_conversation_path = output_dir / "per_conversation_descriptives.csv"
    summary_path = output_dir / "descriptive_summary.csv"

    per_conversation.to_csv(per_conversation_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Processed {len(per_conversation)} conversation files.")
    print(f"Saved per-conversation results to: {per_conversation_path}")
    print(f"Saved summary results to: {summary_path}")

    if skipped:
        skipped_path = output_dir / "skipped_files.csv"
        pd.DataFrame(skipped).to_csv(skipped_path, index=False)
        print(f"Skipped {len(skipped)} files. See: {skipped_path}")


if __name__ == "__main__":
    main()