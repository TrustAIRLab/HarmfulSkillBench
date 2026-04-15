"""
HarmSkillBench Evaluation Runner

Runs LLM evaluation across skills x conditions x models.
Supports OpenAI-compatible API models and local HF Transformers models.

Concurrency model: API tasks are grouped by provider (base_url) and each
provider group gets its OWN ThreadPoolExecutor with `--workers` slots.
This means workers do NOT interfere across providers: the OpenAI group,
the HF Router group, Gemini, and DeepSeek all run at full concurrency
simultaneously instead of being processed sequentially one provider after
another.

Before running, download the benchmark dataset:
    python3 scripts/download_from_hf.py

This populates data/skills/, data/eval_tasks/reviewed_tasks.jsonl, and
data/eval_results/judgments_aggregated.csv.

Usage:
    python3 eval/run_eval.py                                    # Run all (API models)
    python3 eval/run_eval.py --models gpt-4o                    # Single model
    python3 eval/run_eval.py --conditions A B                   # Specific conditions
    python3 eval/run_eval.py --limit 5                          # First 5 skills
    python3 eval/run_eval.py --models llama-3.1-8b --limit 2    # Local model test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from prompts import build_messages, find_skill_md

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REVIEWED_FILE = PROJECT_ROOT / "data" / "eval_tasks" / "reviewed_tasks.jsonl"
RESULTS_DIR = PROJECT_ROOT / "data" / "eval_results"

# ---------------------------------------------------------------------------
# Model configs
#   backend: "openai" (API) or "hf" (HuggingFace Transformers local)
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "gpt-4o": {
        "backend": "openai",
        "model_id": "gpt-4o",
        "use_tool_call": True,
        "extra_body": {},
    },
    "gpt-5.4-mini": {
        "backend": "openai",
        "model_id": "gpt-5.4-mini-2026-03-17",
        "use_tool_call": True,
        "extra_body": {},  # model defaults to reasoning_tokens=0 when unset
    },
    "qwen3-235b": {
        "backend": "openai",
        "model_id": "Qwen/Qwen3-235B-A22B-Instruct-2507:together",
        "base_url": "https://router.huggingface.co/v1",
        "api_key_env": "HF_TOKEN",
        "use_tool_call": True,
        "extra_body": {"thinking": {"type": "disabled"}},
    },
    "gemini-3-flash": {
        "backend": "openai",
        "model_id": "gemini-3-flash-preview",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "use_tool_call": True,
        "extra_body": {"reasoning_effort": "minimal"},  # lowest Gemini 3 can go
    },
    "deepseek-v3.2": {
        "backend": "openai",
        "model_id": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "use_tool_call": True,
        "extra_body": {},
    },
    "kimi-k2.5": {
        "backend": "openai",
        "model_id": "moonshotai/Kimi-K2.5",
        "base_url": "https://router.huggingface.co/v1",
        "api_key_env": "HF_TOKEN",
        "use_tool_call": True,
        "extra_body": {"thinking": {"type": "disabled"}},
    },
    "llama-3.1-8b": {
        "backend": "hf",
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "use_tool_call": False,  # 8B model outputs tool calls instead of text; use fallback
    },
    "qwen3-8b": {
        "backend": "hf",
        "model_id": "Qwen/Qwen3-8B",
        "use_tool_call": False,
        "enable_thinking": False,
    },
}

ALL_CONDITIONS = ["A", "B", "C1", "C2", "C3", "C4", "D"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_reviewed() -> list[dict]:
    """Load reviewed tasks from JSONL."""
    if not REVIEWED_FILE.exists():
        raise FileNotFoundError(
            f"Missing {REVIEWED_FILE}. Run `python scripts/download_from_hf.py` first."
        )
    with REVIEWED_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_result_path(model_name: str, condition: str, anon_id: str) -> Path:
    return RESULTS_DIR / model_name / condition / f"{anon_id}.json"


def result_exists(model_name: str, condition: str, anon_id: str) -> bool:
    """Return True only if a prior successful result exists.
    Errored results (error field set, or empty response_text with no tool call)
    should be retried on re-runs, not skipped.
    """
    path = get_result_path(model_name, condition, anon_id)
    if not path.exists():
        return False
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False  # corrupt → retry
    if data.get("error"):
        return False
    if not (data.get("response_text") or "").strip():
        return False
    return True


def save_result(model_name: str, condition: str, anon_id: str, data: dict) -> None:
    path = get_result_path(model_name, condition, anon_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# OpenAI API backend
# ---------------------------------------------------------------------------

def call_openai(client, config: dict, messages: list, tools=None) -> dict:
    """Call OpenAI-compatible API. All models use temperature=0."""
    kwargs = {
        "model": config["model_id"],
        "messages": messages,
        "temperature": 0,
    }
    if tools:
        kwargs["tools"] = tools
    extra = dict(config.get("extra_body", {}))
    if extra:
        kwargs["extra_body"] = extra

    try:
        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        text = choice.message.content or ""

        # If response is empty but model made tool calls, serialize them as text
        # so the judge can evaluate the intent.
        if not text.strip() and choice.message.tool_calls:
            tc_parts = []
            for tc in choice.message.tool_calls:
                tc_parts.append(
                    f"[Tool Call] {tc.function.name}({tc.function.arguments})"
                )
            text = "\n".join(tc_parts)

        # Some providers may omit `usage`; never crash on a missing field.
        usage_info = None
        if getattr(response, "usage", None) is not None:
            try:
                usage_info = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(response.usage, "completion_tokens", None),
                }
            except Exception:
                usage_info = None

        return {
            "response_text": text,
            "finish_reason": choice.finish_reason,
            "usage": usage_info,
            "error": None,
        }
    except Exception as e:
        return {
            "response_text": "",
            "finish_reason": "error",
            "usage": None,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# HuggingFace Transformers backend
# ---------------------------------------------------------------------------

_hf_models: dict = {}  # cache: model_id -> (model, tokenizer)


def load_hf_model(model_id: str):
    if model_id in _hf_models:
        return _hf_models[model_id]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading HF model: {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _hf_models[model_id] = (model, tokenizer)
    print(f"  Model loaded: {model_id}")
    return model, tokenizer


def call_hf(config: dict, messages: list, tools=None) -> dict:
    """Call local HF model. Uses apply_chat_template with tools support."""
    import torch

    model_id = config["model_id"]

    try:
        model, tokenizer = load_hf_model(model_id)

        template_kwargs = {
            "add_generation_prompt": True,
            "tokenize": False,
        }
        if tools:
            template_kwargs["tools"] = [t["function"] for t in tools]
        if config.get("enable_thinking") is False:
            template_kwargs["enable_thinking"] = False

        prompt_text = tokenizer.apply_chat_template(messages, **template_kwargs)

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=8192,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_ids = outputs[0][input_len:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        return {
            "response_text": response_text,
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": input_len,
                "completion_tokens": len(generated_ids),
            },
            "error": None,
        }
    except Exception as e:
        return {
            "response_text": "",
            "finish_reason": "error",
            "usage": None,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Unified call
# ---------------------------------------------------------------------------

def call_model(client, config: dict, messages: list, tools=None) -> dict:
    if config["backend"] == "openai":
        return call_openai(client, config, messages, tools)
    elif config["backend"] == "hf":
        return call_hf(config, messages, tools)
    return {
        "response_text": "",
        "finish_reason": "error",
        "usage": None,
        "error": f"Unknown backend: {config['backend']}",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run HarmSkillBench evaluation")
    parser.add_argument(
        "--models", nargs="+",
        default=["gpt-4o", "gpt-5.4-mini", "gemini-3-flash",
                 "qwen3-235b", "kimi-k2.5", "deepseek-v3.2"],
        choices=list(MODEL_CONFIGS.keys()),
        help="Models to evaluate",
    )
    parser.add_argument(
        "--conditions", nargs="+", default=["A", "B"],
        choices=ALL_CONDITIONS,
        help="Conditions to run. C1-C4 are Tier 2 2x2 safeguard ablation "
             "(HiTL x AID); D is the no-skill baseline.",
    )
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to first N skills (0=all)")
    parser.add_argument("--tier", choices=["tier1", "tier2"], default=None,
                        help="Filter by tier (default=all)")
    parser.add_argument(
        "--workers", type=int, default=6,
        help="Parallel workers PER PROVIDER GROUP (default=6). Providers are "
             "grouped by base_url (openai / hf_router / gemini / deepseek).",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    skills = load_reviewed()
    if args.tier:
        skills = [s for s in skills if s["tier"] == args.tier]
    if args.limit:
        skills = skills[: args.limit]

    print(f"Skills: {len(skills)}")
    print(f"Models: {args.models}")
    print(f"Conditions: {args.conditions}")

    # Create OpenAI clients for API models (one per unique base_url)
    from openai import OpenAI
    api_clients: dict = {}
    for model_name in args.models:
        cfg = MODEL_CONFIGS[model_name]
        if cfg["backend"] == "openai":
            base_url = cfg.get("base_url")
            key = base_url or "openai_default"
            if key not in api_clients:
                client_kwargs: dict = {}
                if base_url:
                    client_kwargs["base_url"] = base_url
                api_key_env = cfg.get("api_key_env")
                if api_key_env:
                    import os
                    client_kwargs["api_key"] = os.environ.get(api_key_env, "")
                api_clients[key] = OpenAI(**client_kwargs)

    # Collect all pending tasks
    pending_api: list = []
    pending_hf: list = []
    skipped = 0

    for model_name in args.models:
        cfg = MODEL_CONFIGS[model_name]
        for condition in args.conditions:
            for skill in skills:
                anon_id = skill["anon_id"]
                if result_exists(model_name, condition, anon_id):
                    skipped += 1
                    continue
                if cfg["backend"] == "openai":
                    base_url = cfg.get("base_url")
                    client = api_clients[base_url or "openai_default"]
                    pending_api.append((model_name, condition, skill, cfg, client))
                else:
                    pending_hf.append((model_name, condition, skill, cfg))

    total = len(pending_api) + len(pending_hf)
    print(f"Pending: {total} (API={len(pending_api)}, HF={len(pending_hf)}), "
          f"Skipped: {skipped}, Workers: {args.workers}")
    print()

    if total == 0:
        print("Nothing to do.")
        return

    import concurrent.futures
    import threading

    done_count = 0
    error_count = 0
    lock = threading.Lock()

    def process_one(item, is_api=True):
        nonlocal done_count, error_count

        if is_api:
            model_name, condition, skill, cfg, client = item
        else:
            model_name, condition, skill, cfg = item
            client = None

        anon_id = skill["anon_id"]
        # Condition D is a baseline with no skill context — skip the SKILL.md load
        if condition == "D":
            skill_md = ""
        else:
            skill_md = find_skill_md(
                platform=skill["platform"],
                anon_id=anon_id,
                category=skill["category"],
                name=skill["name"],
            )
            if skill_md is None:
                with lock:
                    error_count += 1
                return

        task = skill.get("selected_task", "")
        messages, tools = build_messages(
            skill_name=skill["name"],
            skill_content=skill_md,
            condition=condition,
            task=task,
            use_tool_call=cfg["use_tool_call"],
        )

        result = call_model(client, cfg, messages, tools)

        # Retry up to 3 times on error with exponential backoff (API only).
        if is_api:
            for backoff in (3, 9, 27):
                if not result["error"]:
                    break
                time.sleep(backoff)
                result = call_model(client, cfg, messages, tools)

        output = {
            "anon_id": anon_id,
            "name": skill["name"],
            "category": skill["category"],
            "tier": skill["tier"],
            "condition": condition,
            "model": model_name,
            "task": task if condition != "A" else "",
            "response_text": result["response_text"],
            "finish_reason": result["finish_reason"],
            "usage": result["usage"],
            "error": result["error"],
        }
        save_result(model_name, condition, anon_id, output)

        with lock:
            done_count += 1
            if result["error"]:
                error_count += 1
            if done_count % 20 == 0 or done_count == total:
                status = "OK" if not result["error"] else "ERR"
                print(
                    f"  [{done_count}/{total}] {model_name}/{condition} | "
                    f"{skill['name']} | {status}"
                )

    # Run API models in parallel with a SEPARATE executor per provider group.
    if pending_api:
        from collections import defaultdict

        def provider_key_for(cfg):
            return cfg.get("base_url") or "openai_default"

        pending_by_provider = defaultdict(list)
        for item in pending_api:
            cfg_in_item = item[3]
            pending_by_provider[provider_key_for(cfg_in_item)].append(item)

        print(
            f"Running {len(pending_api)} API calls grouped by "
            f"{len(pending_by_provider)} provider(s), "
            f"{args.workers} workers per provider "
            f"(total max concurrency = {len(pending_by_provider) * args.workers}):"
        )
        for pkey, items in pending_by_provider.items():
            models_in_group = sorted(set(it[0] for it in items))
            print(f"  {pkey}: {len(items)} calls ({', '.join(models_in_group)})")

        provider_executors: dict = {}
        all_futures: list = []
        try:
            for pkey, items in pending_by_provider.items():
                short_name = pkey.replace("https://", "").replace("/", "_")[:24]
                ex = concurrent.futures.ThreadPoolExecutor(
                    max_workers=args.workers,
                    thread_name_prefix=f"prov-{short_name}",
                )
                provider_executors[pkey] = ex
                for item in items:
                    all_futures.append(ex.submit(process_one, item, True))

            concurrent.futures.wait(all_futures)

            for fut in all_futures:
                if fut.exception():
                    print(f"  [ERROR] {fut.exception()}")
        finally:
            for ex in provider_executors.values():
                ex.shutdown(wait=False)

    # Run HF models sequentially (GPU can't parallelize simply).
    if pending_hf:
        print(f"\nRunning {len(pending_hf)} HF calls sequentially...")
        for item in pending_hf:
            process_one(item, is_api=False)

    print(
        f"\nDone. {done_count} completed, {skipped} skipped (checkpoint), "
        f"{error_count} errors."
    )


if __name__ == "__main__":
    main()
