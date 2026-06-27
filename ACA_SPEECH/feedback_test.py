import argparse
import asyncio
import pandas as pd

from setup.constants import prompts, MODEL
from LLM.llm_calls import make_call


def conversation_from_csv(csv_path: str) -> str:
    df = pd.read_csv(csv_path)
    df = df[df["role"].isin(["user", "bot", "assistant"])].copy()

    role_map = {
        "user": "Student",
        "bot": "Examiner",
        "assistant": "Examiner",
    }

    lines = []

    for _, row in df.iterrows():
        role = role_map.get(row["role"], row["role"])
        content = str(row["content"]).strip()

        if content and content.lower() != "nan":
            lines.append(f"{role}: {content}")

    return "\n\n".join(lines)


async def generate_feedback_from_csv(
    csv_path: str,
    llm: str = "gpt",
    max_tokens: int = 400,
) -> tuple[str, int]:
    conversation = conversation_from_csv(csv_path)

    feedback, tokens = await make_call(
        system_prompt=prompts["feedback_prompt"],
        user_prompt=conversation,
        llm=llm,
        max_tokens=max_tokens,
    )

    return feedback, tokens

async def main():
    feedback, tokens = await generate_feedback_from_csv(
        csv_path="/Users/JulieB/PycharmProjects/test_prompts_state/exports/voice_assistant_user_3249_20260524_151248.csv",
        llm=MODEL,
        max_tokens=400,
    )

    print(feedback)
    print(f"\n[feedback_tokens={tokens}]")


if __name__ == "__main__":
    asyncio.run(main())