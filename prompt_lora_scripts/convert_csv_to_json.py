import pandas as pd, json

df = pd.read_csv("summarized_dataset/all_summaries.csv", encoding='utf-8')
records = []
for _, row in df.iterrows():
    records.append({
        "instruction": "Summarize the following text:",
        "input": row["original_text"],
        "output": row["summarized_text"],
    })

with open("summarized_dataset/train.jsonl", "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
