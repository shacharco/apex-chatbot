import json
import re
from time import sleep
from typing import List, Tuple

from ragas import evaluate
from ragas.metrics import answer_correctness, answer_relevancy, faithfulness, context_precision, context_recall
from datasets import Dataset
from langchain_anthropic import ChatAnthropic

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


def run_chatbot(bot, question: str) -> Tuple[str, List[str], List[str], List[str]]:
    result = bot.respond(question)
    # The GeneratorNode returns structured JSON output with "answer"
    generated = result["generated"]
    answer = generated['answer']
    sources = generated['sources']
    retrieved = result["retrieved"]
    categories = result["categories"]
    return answer, sources, retrieved, categories


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
    index = pred.index(true) if true in pred else -1
    return precision, recall, index

def categories_percision_recall(pred: List[str], true: str):
    pred_set = set(pred)
    true_set = {true}
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def evaluate_with_ragas(question: str, answer_pred: str, answer_true: str, contexts: List[str]):
    data = {
        "question": [question],
        "answer": [answer_pred],
        "ground_truth": [answer_true],
        "contexts": [contexts]
    }
    dataset = Dataset.from_dict(data)

    # Configure Anthropic LLM for RAGAS evaluation

    result = evaluate(
        dataset=dataset,
        metrics=[answer_correctness],
    )

    return {
        "answer_correctness": result["answer_correctness"],
        # "faithfulness": result["faithfulness"],
        # "context_precision": result["context_precision"],
        # "context_recall": result["context_recall"]
    }


def main():
    bot = ChatbotGraph(model="gpt-4.1")  # initialize once
    data = load_validation_data()
    total_cp, total_cr = 0.0, 0.0
    total_rp, total_rr = 0.0, 0.0
    total_p, total_r, total_f1 = 0.0, 0.0, 0.0
    total_sp, total_sr = 0.0, 0.0
    total_ar, total_f, total_ctx_p, total_ctx_r = 0.0, 0.0, 0.0, 0.0
    for q, a_true, s_true in data:
        c_true = s_true.split("/")[0]
        a_pred, s_pred, r_pred, c_pred = run_chatbot(bot, q)
        p, r, f1 = precision_recall_f1(a_pred, a_true)
        sp, sr, sr_index = sources_percision_recall(s_pred, s_true)
        rp, rr, rr_index = sources_percision_recall([doc['doc_id'] for doc in r_pred], s_true)

        cp, cr = categories_percision_recall(c_pred, c_true)

        # RAGAS evaluation
        contexts = [doc.get('content', '') for doc in r_pred]
        ragas_scores = evaluate_with_ragas(q, a_pred, a_true, contexts)

        total_cp += cp
        total_cr += cr
        total_rp += rp
        total_rr += rr
        total_sp += sp
        total_sr += sr
        total_p += p
        total_r += r
        total_f1 += f1
        total_ar += ragas_scores["answer_correctness"][0]
        # total_f += ragas_scores["faithfulness"]
        # total_ctx_p += ragas_scores["context_precision"]
        # total_ctx_r += ragas_scores["context_recall"]

        print(f"Q: {q}\nPred: {a_pred} {s_pred}\n"
              f"Gold: {a_true} {s_true}\n"
              f"P={p:.2f} R={r:.2f} F1={f1:.2f} SP={sp:.2f} SR={sr:.2f} RP={rp:.2f} RR={rr:.2f} RI={rr_index:.2f} CP={cp:.2f} CR={cr:.2f}\n"
              f"RAGAS - AC={ragas_scores['answer_correctness'][0]:.2f}\n")
        sleep(1)
    n = len(data)
    if n > 0:
        print(f"AVERAGE Precision={total_p/n:.2f} Recall={total_r/n:.2f} F1={total_f1/n:.2f}"
              f"\nSources Precision={total_sp/n:.2f} Sources Recall={total_sr/n:.2f}\n"
              f"Retrival Precision={total_rp/n:.2f} Retrival Recall={total_rr/n:.2f}\n"
              f"Categories Precision={total_cp/n:.2f} Categories Recall={total_cr/n:.2f}\n"
              f"RAGAS AVERAGES - Answer Relevancy={total_ar/n:.2f} Faithfulness={total_f/n:.2f} Context Precision={total_ctx_p/n:.2f} Context Recall={total_ctx_r/n:.2f}")


if __name__ == "__main__":
    main()