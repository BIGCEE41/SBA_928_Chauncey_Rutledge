import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from rouge_score import rouge_scorer

ART = Path("artifacts")


def load(name):
    rows = [json.loads(line) for line in (ART / name).read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["id"]: r for r in rows}


base = load("base_variation_outputs.jsonl")
tuned = load("finetuned_variation_outputs.jsonl")
prompts = {
    v["id"]: v
    for v in json.loads(Path("data/market_research_prompt_variations.json").read_text(encoding="utf-8"))
}

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
FORMAT_MARKERS = ["key findings", "evidence needed", "bias or limits", "recommendation"]


def words(text):
    return len(text.split())


def format_hits(text):
    t = text.lower()
    return sum(m in t for m in FORMAT_MARKERS)


rows = []
for pid, b in base.items():
    t = tuned[pid]
    p = prompts[pid]
    rows.append({
        "id": pid,
        "style": b["style"],
        "task_type": b["task_type"],
        "industry": b["industry"],
        "region": b["region"],
        "base_words": words(b["output"]),
        "tuned_words": words(t["output"]),
        "base_rougeL_vs_prompt": round(scorer.score(p["prompt"], b["output"])["rougeL"].fmeasure, 4),
        "tuned_rougeL_vs_prompt": round(scorer.score(p["prompt"], t["output"])["rougeL"].fmeasure, 4),
        "base_format_hits": format_hits(b["output"]),
        "tuned_format_hits": format_hits(t["output"]),
        "outputs_identical": b["output"].strip() == t["output"].strip(),
        "base_output": b["output"],
        "tuned_output": t["output"],
        "manual_base_1to5": "",
        "manual_tuned_1to5": "",
    })

with (ART / "variation_scores.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)


def summarize(group_key):
    groups = defaultdict(list)
    for r in rows:
        groups[r[group_key]].append(r)
    out = {}
    for k, rs in sorted(groups.items()):
        out[k] = {
            "n": len(rs),
            "base_avg_words": round(mean(r["base_words"] for r in rs), 1),
            "tuned_avg_words": round(mean(r["tuned_words"] for r in rs), 1),
            "base_avg_rougeL": round(mean(r["base_rougeL_vs_prompt"] for r in rs), 4),
            "tuned_avg_rougeL": round(mean(r["tuned_rougeL_vs_prompt"] for r in rs), 4),
            "base_avg_format_hits": round(mean(r["base_format_hits"] for r in rs), 2),
            "tuned_avg_format_hits": round(mean(r["tuned_format_hits"] for r in rs), 2),
        }
    return out


summary = {
    "total_prompts": len(rows),
    "identical_outputs": sum(r["outputs_identical"] for r in rows),
    "overall": {
        "base_avg_words": round(mean(r["base_words"] for r in rows), 1),
        "tuned_avg_words": round(mean(r["tuned_words"] for r in rows), 1),
        "base_avg_rougeL": round(mean(r["base_rougeL_vs_prompt"] for r in rows), 4),
        "tuned_avg_rougeL": round(mean(r["tuned_rougeL_vs_prompt"] for r in rows), 4),
        "base_avg_format_hits": round(mean(r["base_format_hits"] for r in rows), 2),
        "tuned_avg_format_hits": round(mean(r["tuned_format_hits"] for r in rows), 2),
    },
    "by_style": summarize("style"),
    "by_task_type": summarize("task_type"),
    "by_region": summarize("region"),
    "by_industry": summarize("industry"),
}
(ART / "variation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps({k: summary[k] for k in ("total_prompts", "identical_outputs", "overall")}, indent=2))
print("saved artifacts/variation_scores.csv and artifacts/variation_summary.json")