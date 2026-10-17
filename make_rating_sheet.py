import csv
import random

rows = list(csv.DictReader(open("artifacts/variation_scores.csv", encoding="utf-8")))
diff = [r for r in rows if r["outputs_identical"] == "False"]
random.seed(928)
sample = random.sample(diff, 20)

fields = ["id", "style", "region", "base_output", "tuned_output",
          "manual_base_1to5", "manual_tuned_1to5", "notes"]
with open("artifacts/manual_rating_sheet.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in sample:
        w.writerow({k: r.get(k, "") for k in fields})
print(f"saved artifacts/manual_rating_sheet.csv with {len(sample)} rows")