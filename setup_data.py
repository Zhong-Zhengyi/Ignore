import argparse
import os
import subprocess
from huggingface_hub import snapshot_download


def download_eval_data():
    snapshot_download(
        repo_id="open-unlearning/eval",
        allow_patterns="*.json",
        repo_type="dataset",
        local_dir="saves/eval",
    )


def download_idk_data():
    snapshot_download(
        repo_id="open-unlearning/idk",
        allow_patterns="*.jsonl",
        repo_type="dataset",
        local_dir="data",
    )


def main():
    parser = argparse.ArgumentParser(description="Download and setup evaluation data.")
    parser.add_argument(
        "--eval_logs",
        action="store_true",
        help="Downloads TOFU, MUSE  - retain and finetuned models eval logs and saves them in saves/eval",
    )
    parser.add_argument(
        "--idk",
        action="store_true",
        help="Download idk dataset from HF hub and stores it data/idk.jsonl",
    )

    args = parser.parse_args()
    args.eval_logs = True
    args.idk = False
    args.wmdp = False

    if args.eval_logs:
        download_eval_data()
    if args.idk:
        download_idk_data()

if __name__ == "__main__":
    main()