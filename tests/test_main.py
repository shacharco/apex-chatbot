import json
import re
from time import sleep
from typing import List, Tuple

from chatbot import ChatbotGraph


def load_validation_data(path="tests/validation-data.txt"):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        q, a, s = None, None, None
        for line in f:
            line = line.strip()
            if line.startswith("שאלה:"):
                q = line[5:].strip()
            elif line.startswith("תשובה:"):
                a = line[6:].strip()
            elif line.startswith("מקור:"):
                s = line[5:].strip()
                if q and a and s:
                    pairs.append((q, a, s))
                    q, a, s = None, None, None
    return pairs


def run_chatbot(bot, question: str) -> Tuple[str, List[str]]:
    result = bot.respond(question)
    # The GeneratorNode returns structured JSON output with "answer"
    try:
        generated = result["generated"]
        answer = generated['answer']
        sources = generated['sources']
        return answer, sources
    except Exception:
        return result.get("answer", "")


def precision_recall_f1(pred: str, gold: str):
    pred_tokens = set(re.findall(r"\w+|[^\w\s]", pred.lower(), flags=re.UNICODE))
    gold_tokens = set(re.findall(r"\w+|[^\w\s]", gold.lower(), flags=re.UNICODE))
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0
    tp = len(pred_tokens & gold_tokens)
    fp = len(pred_tokens - gold_tokens)
    fn = len(gold_tokens - pred_tokens)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def sources_percision_recall(pred: List[str], true: str):
    true = ".".join(true.split(",")[0].strip().split('/')[-1].split('.')[:-1])
    pred = [p.split(':')[0] for p in pred]
    pred_set = set(pred)
    true_set = {true}
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return recall, precision


def main():
    bot = ChatbotGraph()  # initialize once
    data = load_validation_data()
    total_p, total_r, total_f1 = 0.0, 0.0, 0.0
    total_sp, total_sr = 0.0, 0.0
    for q, a_true, s_true in data:
        a_pred, s_pred = run_chatbot(bot, q)
        p, r, f1 = precision_recall_f1(a_pred, a_true)
        sp, sr = sources_percision_recall(s_pred, s_true)
        total_sp += sp
        total_sr += sr
        total_p += p
        total_r += r
        total_f1 += f1
        print(f"Q: {q}\nPred: {a_pred} {s_pred}\nGold: {a_true} {s_true}\nP={p:.2f} R={r:.2f} F1={f1:.2f} SP={sp:.2f} SR={sr:.2f}\n")
        sleep(1)
    n = len(data)
    if n > 0:
        print(f"AVERAGE Precision={total_p/n:.2f} Recall={total_r/n:.2f} F1={total_f1/n:.2f} Sources Precision={total_sp/n:.2f} Sources Recall={total_sr/n:.2f}")


if __name__ == "__main__":
    main()