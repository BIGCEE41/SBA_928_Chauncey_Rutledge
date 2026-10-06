# SBA 928: Market-Research Prompt Engineering and Fine-Tuning

## Executive summary

This project evaluates whether a small LoRA fine-tune can improve consistent extraction of customer-voice fields from a public customer-support ticket dataset. It does not treat tickets as a representative market survey and does not infer market prevalence, purchase intent, or causation from individual complaints.

## Dataset and privacy

- **Source:** [Kaggle Customer Support Ticket Dataset](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset), downloaded at runtime with KaggleHub.
- **Use:** individual support tickets as qualitative customer-voice records and recorded category/priority/channel/satisfaction fields as structured labels.
- **Processing:** emails and phone-number-like strings in ticket text are redacted; customer names, email columns, age, and gender are not model inputs. Ticket text is explicitly treated as untrusted data, not instructions. Age and gender are used only in a small-group-suppressed descriptive audit.
- **Representativeness:** support tickets are self-selected, channel-dependent, and not a probability sample. They cannot establish the frequency of a concern among all customers.
- **Source quality:** sampled text contained unresolved template placeholders and boilerplate, and the source does not establish that descriptions are authentic customer statements. Treat this as a pipeline demonstration, not validated primary market research.
- **License:** follow the terms on the source dataset page; raw source data is not redistributed by this project.

The selected sample contains 30 records (24 train and 6 validation tickets), expanded to 144 training examples and 6 held-out examples. The ticket-level split was made before prompt variants were created.

## Prompt engineering

Six prompt variants cover structured issue/product coding, compact analyst output, rating interpretation, evidence-bounded research briefs, JSON output, and a cautious privacy-aware analyst response. Each requests only fields observable in the supplied ticket and instructs against population-level or causal claims. Prompt definitions are in `data/prompt_variations.json`; generated training and validation records are in `artifacts/`. Age and gender are excluded from both prompts and training examples.

## Model and training

- **Base model:** [`google/flan-t5-small`](https://huggingface.co/google/flan-t5-small) (Apache-2.0 per the model card).
- **Method:** parameter-efficient LoRA adaptation; the base weights are not rewritten.
- **Run:** 1 epoch(s), batch size 2, learning rate 0.0002, seed 928, device `cpu`.
- **Mean training loss:** 2.025007.
- **Saved adapter:** `artifacts\fine_tuned_adapter`.

## Base versus fine-tuned results

Both models received the same 6 held-out prompts and used deterministic decoding. Exact match is a strict format-and-string metric, not a complete measure of research quality.

| Model | Exact matches | Exact-match rate |
|---|---:|---:|
| Base | 0/6 | 0.0% |
| Fine-tuned | 0/6 | 0.0% |

The models tied on strict exact match in this small holdout; the run provides no measured exact-match improvement.

Review `artifacts/model_comparison.csv` and the paired output JSONL files before drawing qualitative conclusions. Do not claim improvement unless the held-out results and manual review support it.

## Bias assessment, fairness, and mitigation

`artifacts/fairness_assessment.json` reports descriptive satisfaction summaries by recorded gender and age band when a group has at least five records. This is a screening assessment only: demographic data can be missing, inaccurate, or imbalanced, and satisfaction differences do not establish unfair treatment. The small held-out set does not certify subgroup model performance.

Descriptive findings in this selected sample:

- **gender:** not reported: n=30, mean satisfaction=2.42
- **age_band:** 18-29: suppressed (<5 records), 30-44: n=6, mean satisfaction=1.67, 45-59: n=15, mean satisfaction=2.86, 60+: n=6, mean satisfaction=1.0

Mitigations implemented: exclude direct identifiers and demographics from model prompts; redact emails and phone-like strings; suppress demographic groups with fewer than five records; preserve an explicit non-representativeness caveat in generated briefs; and retain a record-level validation split rather than splitting prompt variants across train and test.

Further work before operational use: expand and independently validate the sample; report subgroup error rates with adequate sample sizes and confidence intervals; have diverse human reviewers assess coding quality and harmful stereotypes; test for missingness and channel/product coverage; and require human review before business decisions.

## Reproducibility and evidence

Run `uv sync`, then `uv run python train.py --max-records 30 --epochs 1`. The command downloads the public dataset, writes the train/validation JSONL files, fine-tunes the adapter, generates paired base/fine-tuned outputs, and records run metadata in `artifacts/training_evidence.json`. Results in this report are populated by that run; raw tickets and artifacts should be handled under the source dataset's terms.

Software versions: {"kagglehub": "1.0.2", "peft": "0.21.2", "torch": "2.14.1", "transformers": "5.18.0"}
