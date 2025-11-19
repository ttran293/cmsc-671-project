import pandas as pd
import json

df = pd.read_csv("summarized_dataset/all_summaries.csv", encoding="utf-8")

records = []
for _, row in df.iterrows():
    original_text = str(row["original_text"])
    summarized_text = str(row["summarized_text"])

    instruction = "Summarize the following text."

    if not original_text or not summarized_text:
        continue

    records.append({
        "instruction": instruction,
        "input": original_text,
        "output": summarized_text
    })

with open("summarized_dataset/train.jsonl", "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
