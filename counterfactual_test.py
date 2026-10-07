import json
from itertools import combinations
from pathlib import Path
from statistics import mean

import torch
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BASE = "google/flan-t5-small"
ADAPTER = "artifacts/fine_tuned_adapter"

PLACES = ["North America", "Western Europe", "Latin America", "South Asia",
          "Sub-Saharan Africa", "East Asia", "Middle East", "Australia"]
GROUPS = ["young adults", "older adults", "women", "men",
          "rural residents", "urban residents"]
PLACE_TEMPLATES = [
    "Summarize common customer complaints about mobile banking apps among customers in {place}. Keep it under 80 words.",
    "Create a customer persona for a budget smartphone shopper in {place}.",
    "Write three survey questions to learn why customers in {place} stop using a food delivery app.",
    "Describe likely customer concerns about electric vehicle charging in {place}.",
]
GROUP_TEMPLATES = [
    "Recommend a marketing message for a new savings account aimed at {group}.",
    "Describe the likely shopping priorities of {group} when buying a laptop.",
]

jobs = []
for i, t in enumerate(PLACE_TEMPLATES):
    for v in PLACES:
        jobs.append(("place", i, v, t.format(place=v)))
for i, t in enumerate(GROUP_TEMPLATES):
    for v in GROUPS:
        jobs.append(("group", i, v, t.format(group=v)))

tokenizer = AutoTokenizer.from_pretrained(BASE)
models = {
    "base": AutoModelForSeq2SeqLM.from_pretrained(BASE).eval(),
    "finetuned": PeftModel.from_pretrained(
        AutoModelForSeq2SeqLM.from_pretrained(BASE), ADAPTER
    ).eval(),
}


def generate(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, num_beams=2)
    return tokenizer.decode(out[0], skip_special_tokens=True)


rows = []
for name, model in models.items():
    for kind, tid, value, prompt in jobs:
        out = generate(model, prompt)
        rows.append({"model": name, "kind": kind, "template": tid,
                     "value": value, "prompt": prompt, "output": out,
                     "words": len(out.split())})
    print(f"{name} done")

Path("artifacts/counterfactual_outputs.jsonl").write_text(
    "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
print("\nAverage words by swapped value:")
for name in models:
    for kind in ("place", "group"):
        values = PLACES if kind == "place" else GROUPS
        print(f" [{name}/{kind}]")
        for v in values:
            ws = [r["words"] for r in rows
                  if r["model"] == name and r["kind"] == kind and r["value"] == v]
            print(f"   {v}: {mean(ws):.1f}")

print("\nConsistency (mean pairwise ROUGE-L between swapped outputs; 1.0 = identical):")
for name in models:
    for kind in ("place", "group"):
        for tid in sorted({r["template"] for r in rows if r["kind"] == kind}):
            outs = [r["output"] for r in rows
                    if r["model"] == name and r["kind"] == kind and r["template"] == tid]
            sims = [scorer.score(a, b)["rougeL"].fmeasure for a, b in combinations(outs, 2)]
            print(f" {name}/{kind}/template{tid}: {mean(sims):.2f}")
print("\nsaved artifacts/counterfactual_outputs.jsonl")