import json
import re
from typing import Dict

from chatbot import graph


def load_validation_data(path="validation-data.txt"):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        q, a = None, None
        for line in f:
            line = line.strip()
            if line.startswith("q:"):
                q = line[2:].strip()
            elif line.startswith("a:"):
                a = line[2:].strip()
                if q and a:
                    pairs.append((q, a))
                    q, a = None, None
    return pairs


def run_chatbot(question: str) -> str:
    state = {"question": question}
    result = graph.invoke(state)
    # The GeneratorNode returns structured JSON output with "answer"
    try:
        parsed = json.loads(result["answer"])
        return parsed.get("answer", "")
    except Exception:
        return result.get("answer", "")


def precision_recall_f1(pred: str, gold: str):
    pred_tokens = set(re.findall(r"\\w+", pred.lower()))
    gold_tokens = set(re.findall(r"\\w+", gold.lower()))
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0
    tp = len(pred_tokens & gold_tokens)
    fp = len(pred_tokens - gold_tokens)
    fn = len(gold_tokens - pred_tokens)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def main():
    data = load_validation_data()
    total_p, total_r, total_f1 = 0.0, 0.0, 0.0
    for q, gold in data:
        pred = run_chatbot(q)
        p, r, f1 = precision_recall_f1(pred, gold)
        total_p += p
        total_r += r
        total_f1 += f1
        print(f"Q: {q}\nPred: {pred}\nGold: {gold}\nP={p:.2f} R={r:.2f} F1={f1:.2f}\n")
    n = len(data)
    if n > 0:
        print(f"AVERAGE Precision={total_p/n:.2f} Recall={total_r/n:.2f} F1={total_f1/n:.2f}")


if __name__ == "__main__":
    main()