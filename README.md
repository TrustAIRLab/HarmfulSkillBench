# HarmSkillBench

Benchmark code for evaluating LLM refusal behavior when agents are exposed
to skills that describe potentially harmful capabilities.

The benchmark probes whether current LLMs can detect and refuse harmful
agent skills in two settings. **Tier 1** covers prohibited behaviors that
should always be refused. **Tier 2** covers high-risk domains where
responses should include human-in-the-loop referral and AI disclosure
safeguards.

- **Dataset (200 skills + tasks + aggregated results)**:
  [TrustAIRLab/HarmSkillBench on HuggingFace](https://huggingface.co/datasets/TrustAIRLab/HarmSkillBench)
  (gated)
- **Code (this repo)**: download script + evaluation runner + judge +
  analysis + figure plotting

## Quick Start

```bash
# 1. Create a conda env and install requirements
conda create -n harmskillbench python=3.11 -y
conda activate harmskillbench
pip install -r requirements.txt

# 2. Configure API keys
cp .env.template .env
# edit .env and fill in the keys you have (HF_TOKEN is required; at minimum
# OPENAI_API_KEY is needed to run the default models and the judge)

# 3. Download the dataset from HuggingFace
python scripts/download_from_hf.py

# 4. Run a tiny smoke test
python eval/run_eval.py  --models gpt-4o --conditions A --limit 1
python eval/judge.py     --models gpt-4o --conditions A --limit 1

# 5. Full benchmark (all 6 models, all conditions, all 200 skills)
python eval/run_eval.py
python eval/judge.py
python eval/analyze.py
python eval/plot_benchmark.py
```

## Dataset Summary

| Split | Count |
|---|---|
| Total skills | 200 |
| Tier 1 Prohibited (A1-A14, A7 excluded) | 130 |
| Tier 2 High-Risk (AH1-AH7) | 70 |
| Source platforms | clawhub (81), skillsrest (57), synthetic (62) |
| Human-reviewed harmful tasks | 200 |
| Evaluated models | 6 |
| Evaluation conditions | 7 (A, B, C1-C4, D) |
| Aggregated judgments | 5,280 rows |

See [DATA.md](DATA.md) for the detailed field schema.

## Evaluation Conditions

| Code | Description | Scope |
|---|---|---|
| A | Passive skill reading | All 200 skills |
| B | Explicit harmful task with skill available | All 200 skills |
| C1-C4 | Tier 2 2x2 safeguard ablation (HiTL x AID instructions) | 70 skills each |
| D | Explicit task without skill content (no-skill baseline) | All 200 skills |

## Reproducing the Published Results

The dataset ships with `eval_results/judgments_aggregated.csv` containing
the 5,280 judgments reported in the paper. Running the evaluation pipeline
above against the same (model, condition, skill) cell reproduces those
judgments (temperature = 0 for both target models and the judge).

A smoke-level reproduction check (gpt-4o / Condition A / first skill)
exactly matches the CSV on `label`, `refusal`, `hitl`, `aid`,
`harmfulness`, and `score`.

## Documentation

- [SETUP.md](SETUP.md): environment, dependencies, API keys
- [EVALUATE.md](EVALUATE.md): full evaluation workflow (download, run,
  judge, analyze, plot)
- [DATA.md](DATA.md): dataset layout and field schema

## License

MIT. See [LICENSE](LICENSE).

## Citation

```bibtex
@misc{harmskillbench2026,
  title  = {HarmSkillBench: A Benchmark for Harmful Agent Skill Refusal},
  author = {TrustAIRLab},
  year   = {2026},
  url    = {https://huggingface.co/datasets/TrustAIRLab/HarmSkillBench},
}
```
