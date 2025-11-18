#!/usr/bin/env python3
"""Script to push a model directory to Hugging Face Hub."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo, upload_folder

# Load environment variables
load_dotenv()


def push_to_hub(
    local_dir: str,
    repo_id: str = None,
    account: str = None,
    private: bool = False,
    token: str = None,
):
    """Push a local model directory to Hugging Face Hub."""
    # Get token from environment if not provided
    if token is None:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    if token is None:
        raise ValueError(
            "HF token not found. Please set HF_TOKEN or HUGGINGFACE_TOKEN environment variable."
        )

    # Determine repo_id from path if not provided
    if repo_id is None:
        if account is None:
            account = os.getenv("HF_ACCOUNT", "bcywinski")

        # Extract model name from path
        path_parts = Path(local_dir).parts
        # Find the model name (e.g., "Llama-3.2-1B-Instruct")
        model_name = None
        for part in path_parts:
            if "Llama" in part or "llama" in part.lower():
                model_name = part
                break

        if model_name is None:
            model_name = path_parts[-1] if path_parts else "model"

        # Create repo_id: account/model-name-config
        repo_id = f"{account}/{model_name.lower().replace('.', '-')}"

    # Create repository if it doesn't exist
    api = HfApi(token=token)
    try:
        create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=True,
        )
        print(f"Repository {repo_id} is ready.")
    except Exception as e:
        print(f"Note: {e}")

    # Upload the folder
    print(f"Uploading {local_dir} to {repo_id}...")
    upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        token=token,
        ignore_patterns=[".git*", "__pycache__", "*.pyc"],
    )
    print(f"Successfully pushed to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push model to Hugging Face Hub")
    parser.add_argument(
        "local_dir",
        type=str,
        help="Path to local model directory",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Repository ID (e.g., 'account/model-name'). If not provided, will be inferred from path.",
    )
    parser.add_argument(
        "--account",
        type=str,
        default=None,
        help="Hugging Face account/organization name",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (or use HF_TOKEN/HUGGINGFACE_TOKEN env var)",
    )

    args = parser.parse_args()

    push_to_hub(
        local_dir=args.local_dir,
        repo_id=args.repo_id,
        account=args.account,
        private=args.private,
        token=args.token,
    )
