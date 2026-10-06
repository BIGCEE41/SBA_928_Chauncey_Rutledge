import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BASE = "google/flan-t5-small"
ADAPTER = "artifacts/fine_tuned_adapter"
VARIATIONS = Path("data/market_research_prompt_variations.json")
OUT_DIR = Path("artifacts")

variations = json.loads(VARIATIONS.read_text(encoding="utf-8"))
tokenizer = AutoTokenizer.from_pretrained(BASE)
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE).eval()
tuned_model = PeftModel.from_pretrained(
    AutoModelForSeq2SeqLM.from_pretrained(BASE), ADAPTER
).eval()


def generate(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, num_beams=2)
    return tokenizer.decode(out[0], skip_special_tokens=True)


for name, model in [("base", base_model), ("finetuned", tuned_model)]:
    path = OUT_DIR / f"{name}_variation_outputs.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for i, v in enumerate(variations, 1):
            row = {
                "id": v["id"],
                "base_id": v["base_id"],
                "style": v["style"],
                "task_type": v["task_type"],
                "industry": v["industry"],
                "region": v["region"],
                "model": name,
                "output": generate(model, v["prompt"]),
            }
            f.write(json.dumps(row) + "\n")
            if i % 24 == 0:
                print(f"{name}: {i}/{len(variations)}")
    print(f"saved {path}")