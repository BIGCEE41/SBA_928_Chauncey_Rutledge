import json
from pathlib import Path

DATA = Path("data")

prompts = json.loads((DATA / "market_research_prompts.json").read_text(encoding="utf-8"))
second = DATA / "market_research_prompts_2.json"
if second.exists():
    prompts += json.loads(second.read_text(encoding="utf-8"))

FEW_SHOT_EXAMPLE = (
    "Example task: Summarize themes in feedback about a coffee shop.\n"
    "Example answer: Three themes appear: (1) slow service at peak hours, "
    "(2) praise for drink quality, (3) confusion about loyalty rewards. "
    "Evidence is limited to the comments provided.\n\n"
)

STYLES = {
    "plain": lambda p: p,
    "role_based": lambda p: (
        "You are an experienced market-research analyst. Be accurate, neutral, "
        "and say when evidence is limited.\n\nTask: " + p
    ),
    "few_shot": lambda p: FEW_SHOT_EXAMPLE + "Now do this task: " + p,
    "structured_cot": lambda p: (
        p + "\n\nThink step by step. Then answer in this format:\n"
        "Key findings: ...\nEvidence needed: ...\nBias or limits: ...\nRecommendation: ..."
    ),
}

variations = []
for item in prompts:
    for style, build in STYLES.items():
        variations.append({
            "id": f"{item['id']}_{style}",
            "base_id": item["id"],
            "style": style,
            "task_type": item["task_type"],
            "industry": item["industry"],
            "region": item["region"],
            "audience": item["audience"],
            "prompt": build(item["prompt"]),
        })

out = DATA / "market_research_prompt_variations.json"
out.write_text(json.dumps(variations, indent=2), encoding="utf-8")
print(f"{len(prompts)} prompts -> {len(variations)} variations saved to {out}")