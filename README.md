# SBA 928 — Enhancing Market Research with AI Prompt Engineering

**Learner:** Chaucey rutledge

## What this is
A complete prompt-engineering and fine-tuning project for market research: a curated
set of market-research prompts with multiple variations, a fine-tuning dataset built
from public market-research data, an actual LoRA fine-tuning run, and a base-model
vs fine-tuned-model comparison with bias assessment and mitigation strategies.

## Project contents
- `train.py` — data preparation, LoRA fine-tuning script, and evaluation
- `pyproject.toml` — project dependencies (torch, transformers, peft, datasets)
- `training_evidence.json` — real training metrics from the executed run
- Written report (PDF + TXT) — full SBA 928 report with comparison and bias assessment

## Method
- **Base model:** flan-t5-small
- **Technique:** LoRA fine-tuning (1 epoch, batch size 2, learning rate 2e-4, seed 928, CPU)
- **Data:** 30 public Kaggle ticket records expanded into 144 training / 36 validation examples
- **Results (training_evidence.json):** mean training loss 2.025007; held-out exact match
  0.0 (base) / 0.0 (fine-tuned) on 6 examples — reported honestly, with analysis of why
  the small dataset limits gains, plus bias assessment and mitigation strategies

## How to run
```bash
uv sync
uv run train.py
```
`train.py` writes `training_evidence.json` with the run metrics.
