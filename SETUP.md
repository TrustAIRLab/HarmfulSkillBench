# SETUP

## 1. Python environment

Create a dedicated conda environment.

```bash
conda create -n harmskillbench python=3.11 -y
conda activate harmskillbench
pip install -r requirements.txt
```

`requirements.txt` declares the minimum runtime dependencies:

```
huggingface_hub>=0.20
openai>=1.40
python-dotenv>=1.0
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.13
```

Running local models (`--models llama-3.1-8b` or `qwen3-8b`) additionally
requires `torch` and `transformers`. Install separately:

```bash
pip install torch transformers accelerate
```

## 2. API keys

Copy `.env.template` to `.env` and fill in the keys for the providers you
intend to use.

| Variable | Used for |
|---|---|
| `HF_TOKEN` | Downloading the gated HuggingFace dataset; also the API key for `qwen3-235b` and `kimi-k2.5` (accessed via HF Router) |
| `OPENAI_API_KEY` | `gpt-4o`, `gpt-5.4-mini`, and the judge model |
| `GEMINI_API_KEY` | `gemini-3-flash` |
| `DEEPSEEK_API_KEY` | `deepseek-v3.2` |

At minimum you need `HF_TOKEN` (to download data) plus `OPENAI_API_KEY`
(for the judge model).

## 3. HuggingFace access

The dataset repo `TrustAIRLab/HarmSkillBench` is gated. Click "Request
Access" on the dataset page and fill out the form. Once the request
is approved, your HF token can download the dataset.

## 4. Download the data

```bash
python scripts/download_from_hf.py
```

This writes the dataset to `data/` with the following structure:

```
data/
  README.md
  LICENSE
  skills/
    clawhub/{anon_id}/{SKILL.md, _meta.json}
    skillsrest/{anon_id}/{SKILL.md, _meta.json}
    synthetic/{category}/{name}/{SKILL.md, _meta.json}
  eval_tasks/
    reviewed_tasks.jsonl
  eval_results/
    judgments_aggregated.csv
```

## 5. Verify the install

```bash
python eval/run_eval.py --models gpt-4o --conditions A --limit 1
python eval/judge.py    --models gpt-4o --conditions A --limit 1
```

This performs 1 API call + 1 judge call and writes a result to
`data/eval_results/gpt-4o/A/` and a judgment to
`data/eval_results/judgments/gpt-4o/A/`.
