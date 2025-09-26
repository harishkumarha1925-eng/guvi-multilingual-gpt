"""
Converts raw GUVI texts (FAQs, course blurbs, support logs) into instruction
tuning JSONL: {"instruction": "...", "input": "", "output": "..."}
"""
import json, csv, argparse, pathlib

def main(in_file, out_file):
    data = []
    # Example: CSV with columns question, answer
    with open(in_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "instruction": "Answer the user's question about GUVI.",
                "input": row["question"],
                "output": row["answer"]
            })
    with open(out_file, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", required=True)
    ap.add_argument("--out_file", default="dataset.jsonl")
    args = ap.parse_args()
    main(args.in_file, args.out_file)
