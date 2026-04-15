"""
Download HarmSkillBench from HuggingFace into the local `data/` directory.

The HuggingFace dataset repo is gated. You need:
  1. A HuggingFace account with access granted to TrustAIRLab/HarmSkillBench
     (apply via the dataset page; approval is automatic for researchers).
  2. An HF token stored in either:
       - the HF_TOKEN environment variable, or
       - a .env file next to this script's repo root

Usage:
    python scripts/download_from_hf.py
    python scripts/download_from_hf.py --revision v1.0      # pin a release
    python scripts/download_from_hf.py --local-dir ./data   # custom destination

After download, the expected layout is:
    data/
      README.md
      LICENSE
      skills/{clawhub,skillsrest,synthetic}/...
      eval_tasks/reviewed_tasks.jsonl
      eval_results/judgments_aggregated.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ID = "TrustAIRLab/HarmSkillBench"
REPO_TYPE = "dataset"
DEFAULT_LOCAL_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    ap = argparse.ArgumentParser(description="Download HarmSkillBench from HuggingFace.")
    ap.add_argument(
        "--local-dir",
        default=str(DEFAULT_LOCAL_DIR),
        help=f"Destination folder (default: {DEFAULT_LOCAL_DIR})",
    )
    ap.add_argument(
        "--revision",
        default=None,
        help="Specific git revision, branch or tag (default: main)",
    )
    ap.add_argument(
        "--token",
        default=None,
        help="HF access token (overrides HF_TOKEN env var and .env)",
    )
    args = ap.parse_args()

    # Load .env next to the repo root if the token is not already set.
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except ModuleNotFoundError:
        pass  # dotenv is optional; token may already be in env

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        print(
            "[error] No HF token found. Set HF_TOKEN in your environment or .env, "
            "or pass --token.",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError:
        print(
            "[error] huggingface_hub is not installed. Run: pip install huggingface_hub",
            file=sys.stderr,
        )
        sys.exit(2)

    local_dir = Path(args.local_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"[plan]  repo:      {REPO_ID}  (type={REPO_TYPE})")
    print(f"[plan]  revision:  {args.revision or 'main'}")
    print(f"[plan]  local_dir: {local_dir}")

    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        revision=args.revision,
        local_dir=str(local_dir),
        token=token,
    )
    print(f"[done]  downloaded to: {path}")

    # Minimal post-download sanity check.
    expected = [
        "skills",
        "eval_tasks/reviewed_tasks.jsonl",
        "eval_results/judgments_aggregated.csv",
        "README.md",
    ]
    missing = [p for p in expected if not (local_dir / p).exists()]
    if missing:
        print(f"[warn]  missing expected paths: {missing}", file=sys.stderr)
        sys.exit(1)
    print("[ok]    all expected paths are present.")


if __name__ == "__main__":
    main()
