from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd


ROLE_USER = "user"
ROLE_BOT = "bot"
ROLE_FEEDBACK = "feedback"
VALID_ROLES = {ROLE_USER, ROLE_BOT, ROLE_FEEDBACK}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract descriptive conversation metrics from ACA-bot CSV files."
    )

    parser.add_argument(
        "--csv-folder",
        required=True,
        help="Folder containing conversation CSV files.",
    )

    parser.add_argument(
        "--logs-json",
        default=None,
        help="Optional JSONL conversation log file used to calculate duration.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Folder where output CSV files will be saved.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search CSV files recursively.",
    )

    parser.add_argument(
        "--require-feedback",
        action="store_true",
        help="Only include conversations that contain a feedback turn.",
    )

    parser.add_argument(
        "--min-turns",
        type=int,
        default=None,
        help="Optional minimum number of turns for inclusion, e.g. 23.",
    )

    return parser.parse_args()


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def word_count(text: str) -> int:
    text = clean_text(text)
    if not text:
        return 0
    return len(re.findall(r"\b[\w']+\b", text))


def parse_timestamp(value: Any) -> Optional[pd.Timestamp]:
    if value is None or pd.isna(value):
        return None
    try:
        return pd.to_datetime(value, utc=True)
    except Exception:
        return None


def extract_username_from_filename(path: Path) -> Optional[str]:
    match = re.search(r"voice_assistant_user_\d+", path.name)
    if match:
        return match.group(0)
    return None


def extract_datetime_from_filename(path: Path) -> Optional[pd.Timestamp]:
    """
    Example filename:
    voice_assistant_user_9271_20260609_120016.csv

    Interprets the timestamp as UTC for matching purposes.
    """
    match = re.search(r"(\d{8})_(\d{6})", path.name)
    if not match:
        return None

    date_part, time_part = match.groups()
    try:
        dt = datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")
        return pd.Timestamp(dt, tz="UTC")
    except Exception:
        return None


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Supports both:
    - JSONL: one JSON object per line
    - JSON: a list of objects
    """
    text = path.read_text(encoding="utf-8").strip()

    if not text:
        return []

    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of events.")
        return data

    events = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        events.append(json.loads(line))

    return events


def get_event_username(event: Dict[str, Any]) -> Optional[str]:
    meta = event.get("meta") or {}

    if isinstance(meta, dict):
        username = meta.get("username")
        if username:
            return str(username)

    author_id = event.get("author_id")
    if author_id and str(author_id).startswith("voice_assistant_user_"):
        return str(author_id)

    return None


def summarize_logs_by_conversation(logs_path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if logs_path is None:
        return {}

    events = load_json_or_jsonl(Path(logs_path))

    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for event in events:
        conversation_id = event.get("conversation_id")
        if not conversation_id:
            continue
        grouped.setdefault(str(conversation_id), []).append(event)

    summaries: Dict[str, Dict[str, Any]] = {}

    for conversation_id, conv_events in grouped.items():
        timestamps = [
            parse_timestamp(event.get("timestamp_server_utc"))
            for event in conv_events
        ]
        timestamps = [ts for ts in timestamps if ts is not None]

        if timestamps:
            start_time = min(timestamps)
            end_time = max(timestamps)
            duration_minutes = (end_time - start_time).total_seconds() / 60
        else:
            start_time = None
            end_time = None
            duration_minutes = None

        usernames = [
            get_event_username(event)
            for event in conv_events
            if get_event_username(event)
        ]
        username = usernames[0] if usernames else None

        event_text = "\n".join(
            json.dumps(event, ensure_ascii=False)
            for event in conv_events
        )

        summaries[conversation_id] = {
            "conversation_id": conversation_id,
            "username": username,
            "start_time": start_time,
            "end_time": end_time,
            "duration_minutes": duration_minutes,
            "n_log_events": len(conv_events),
            "n_log_error_markers": event_text.count("[error]"),
        }

    return summaries


def match_csv_to_log_summary(
    csv_path: Path,
    log_summaries: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not log_summaries:
        return None

    username = extract_username_from_filename(csv_path)
    file_time = extract_datetime_from_filename(csv_path)

    candidates = list(log_summaries.values())

    if username:
        candidates = [
            item for item in candidates
            if item.get("username") == username
        ]

    if not candidates:
        return None

    if file_time is not None:
        candidates_with_time = [
            item for item in candidates
            if item.get("start_time") is not None
        ]

        if candidates_with_time:
            return min(
                candidates_with_time,
                key=lambda item: abs(
                    (item["start_time"] - file_time).total_seconds()
                ),
            )

    if len(candidates) == 1:
        return candidates[0]

    return None


def summarize_one_csv(
    csv_path: Path,
    log_summaries: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)

    if "role" not in df.columns or "content" not in df.columns:
        raise ValueError(f"{csv_path.name} must contain columns 'role' and 'content'.")

    df["role"] = df["role"].apply(clean_text).str.lower()
    df["content"] = df["content"].apply(clean_text)

    df = df[
        df["role"].isin(VALID_ROLES)
        & (df["content"] != "")
    ].copy()

    n_turns = len(df)

    user_df = df[df["role"] == ROLE_USER]
    bot_df = df[df["role"] == ROLE_BOT]
    feedback_df = df[df["role"] == ROLE_FEEDBACK]

    user_word_counts = user_df["content"].apply(word_count)
    bot_word_counts = bot_df["content"].apply(word_count)

    if len(feedback_df) > 0:
        final_feedback_text = feedback_df.iloc[-1]["content"]
        feedback_length_words = word_count(final_feedback_text)
    else:
        feedback_length_words = None

    all_text = "\n".join(
        df.astype(str).agg(" ".join, axis=1).tolist()
    )

    n_error_markers_csv = all_text.count("[error]")

    n_state_parse_errors = 0
    if "debug_state" in df.columns:
        debug_text = "\n".join(df["debug_state"].fillna("").astype(str).tolist())
        n_state_parse_errors = debug_text.count("Could not parse model state JSON")

    matched_log = match_csv_to_log_summary(csv_path, log_summaries)

    if matched_log:
        conversation_id = matched_log.get("conversation_id")
        duration_minutes = matched_log.get("duration_minutes")
        log_start_time = matched_log.get("start_time")
        log_end_time = matched_log.get("end_time")
        n_log_error_markers = matched_log.get("n_log_error_markers")
    else:
        conversation_id = None
        duration_minutes = None
        log_start_time = None
        log_end_time = None
        n_log_error_markers = None

    return {
        "conversation_file": csv_path.name,
        "matched_conversation_id": conversation_id,
        "n_turns": n_turns,
        "n_user_turns": len(user_df),
        "n_examiner_turns": len(bot_df),
        "n_feedback_turns": len(feedback_df),
        "has_feedback": len(feedback_df) > 0,
        "mean_user_response_length_words": user_word_counts.mean(),
        "sd_user_response_length_words": user_word_counts.std(),
        "mean_examiner_response_length_words": bot_word_counts.mean(),
        "sd_examiner_response_length_words": bot_word_counts.std(),
        "feedback_length_words": feedback_length_words,
        "duration_minutes": duration_minutes,
        "log_start_time": log_start_time,
        "log_end_time": log_end_time,
        "n_error_markers_csv": n_error_markers_csv,
        "n_error_markers_logs": n_log_error_markers,
        "n_state_parse_errors": n_state_parse_errors,
    }


def make_overall_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "n_turns",
        "n_user_turns",
        "n_examiner_turns",
        "n_feedback_turns",
        "mean_user_response_length_words",
        "mean_examiner_response_length_words",
        "feedback_length_words",
        "duration_minutes",
        "n_error_markers_csv",
        "n_error_markers_logs",
        "n_state_parse_errors",
    ]

    rows = []

    for metric in metrics:
        if metric not in df.columns:
            continue

        values = pd.to_numeric(df[metric], errors="coerce").dropna()

        if len(values) == 0:
            rows.append({
                "metric": metric,
                "n_conversations": 0,
                "mean": None,
                "sd": None,
                "min": None,
                "max": None,
                "sum": None,
            })
            continue

        rows.append({
            "metric": metric,
            "n_conversations": len(values),
            "mean": values.mean(),
            "sd": values.std(),
            "min": values.min(),
            "max": values.max(),
            "sum": values.sum(),
        })

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    csv_folder = Path(args.csv_folder)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = "**/*.csv" if args.recursive else "*.csv"
    csv_files = sorted(csv_folder.glob(pattern))

    if not csv_files:
        raise ValueError(f"No CSV files found in {csv_folder}")

    log_summaries = summarize_logs_by_conversation(args.logs_json)

    rows = []

    for csv_path in csv_files:
        try:
            row = summarize_one_csv(csv_path, log_summaries)
            row["status"] = "success"
            row["error"] = None
        except Exception as exc:
            row = {
                "conversation_file": csv_path.name,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }

        rows.append(row)

    per_conversation = pd.DataFrame(rows)

    included = per_conversation[per_conversation["status"] == "success"].copy()

    if args.require_feedback:
        included = included[included["has_feedback"] == True]

    if args.min_turns is not None:
        included = included[included["n_turns"] >= args.min_turns]

    overall_summary = make_overall_summary(included)

    per_conversation.to_csv(
        output_dir / "conversation_descriptive_metrics_all.csv",
        index=False,
        encoding="utf-8",
    )

    included.to_csv(
        output_dir / "conversation_descriptive_metrics_included.csv",
        index=False,
        encoding="utf-8",
    )

    overall_summary.to_csv(
        output_dir / "conversation_descriptive_summary.csv",
        index=False,
        encoding="utf-8",
    )

    print(f"Processed {len(csv_files)} CSV file(s).")
    print(f"Included {len(included)} conversation(s) after filters.")
    print(f"Saved all metrics to: {output_dir / 'conversation_descriptive_metrics_all.csv'}")
    print(f"Saved included metrics to: {output_dir / 'conversation_descriptive_metrics_included.csv'}")
    print(f"Saved summary to: {output_dir / 'conversation_descriptive_summary.csv'}")


if __name__ == "__main__":
    main()