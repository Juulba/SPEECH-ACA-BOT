import argparse
import asyncio

from logger import JSONLogger


async def export_logs(log_file, export_dir="exports"):
    logger = JSONLogger(log_file=log_file)

    print(f"Reading conversation logs from: {log_file}")
    print(f"Exporting CSV files to: {export_dir}")

    await logger.export_conversation_csvs(export_dir=export_dir)

    print("Done.")


def parse_args():
    parser = argparse.ArgumentParser(description="Export conversation logs to CSV files.")

    parser.add_argument(
        "--log-file",
        required=True,
        help="Path to the JSON/JSONL conversation log file.",
    )

    parser.add_argument(
        "--export-dir",
        default="exports",
        help="Directory where exported CSV files will be saved.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(export_logs(log_file=args.log_file, export_dir=args.export_dir))


if __name__ == "__main__":
    main()