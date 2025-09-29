# FILE: chatbot.py
"""
LangGraph-based chatbot pipeline.
Nodes: Router -> Retriever -> Generator
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, TypedDict

import faiss
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_mistralai import ChatMistralAI

from langgraph.graph import StateGraph, END

class ChatbotState(TypedDict):
    question: str
    category: str
    retrieved: List[str]
    generated: str


init_state = ChatbotState(
    question="",
    category="",
    retrieved=[],
    generated=""
)

# Router Node
class RouterNode:
    def __init__(self, categories=None):
        self.categories = categories or ["car_insurance", "life_insurance", "travel_insurance"]
        self.kws = {
            "car_insurance": ["car_insurance", "vehicle", "collision", "comprehensive", "auto", "windshield", "license"],
            "life_insurance": ["life", "beneficiary", "death", "term life", "whole life", "premium", "insured"],
            "travel_insurance": ["travel", "trip", "lost luggage", "cancel", "medical abroad", "flight", "delay"],
        }

    def __call__(self, state):
        q = state["question"].lower()
        scores = {c: 0 for c in self.categories}
        for c in self.categories:
            for k in self.kws.get(c, []):
                if k in q:
                    scores[c] += 1
        best = max(scores.items(), key=lambda x: x[1])[0]
        state["category"] = best
        return state


# Retriever Node
class RetrieverNode:
    def __init__(self, indices_dir="indices", k=20):
        self.indices_dir = indices_dir
        self.k = k
        self.loaded = {}

    def _load_category(self, category):
        if category in self.loaded:
            return self.loaded[category]
        base = os.path.join(self.indices_dir, category)
        if not os.path.exists(base):
            raise FileNotFoundError(f"No index for category {category} in {self.indices_dir}")
        vec = joblib.load(os.path.join(base, "tfidf_vectorizer.joblib"))
        docs_meta = pickle.load(open(os.path.join(base, "doc_ids.pkl"), "rb"))
        index = faiss.read_index(os.path.join(base, "faiss.index"))
        self.loaded[category] = {"vec": vec, "docs_meta": docs_meta, "index": index}
        return self.loaded[category]

    def __call__(self, state):
        category = state["category"]
        question = state["question"]
        data = self._load_category(category)
        vec: TfidfVectorizer = data["vec"]
        docs_meta = data["docs_meta"]
        index = data["index"]

        q_vec = vec.transform([question]).astype(np.float32).toarray()
        qn = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-9)
        D, I = index.search(qn, self.k)

        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(docs_meta["docs"]):
                continue
            results.append(
                {"doc_id": docs_meta["doc_ids"][idx], "text": docs_meta["docs"][idx], "score": float(score)}
            )

        # Add TF-IDF top-k
        tfidf_scores = (vec.transform([question]).dot(vec.transform(docs_meta["docs"]).T)).toarray()[0]
        top_tfidf_idx = np.argsort(-tfidf_scores)[: self.k]
        for idx in top_tfidf_idx:
            if idx < 0 or idx >= len(docs_meta["docs"]):
                continue
            did = docs_meta["doc_ids"][idx]
            if not any(r["doc_id"] == did for r in results):
                results.append(
                    {"doc_id": did, "text": docs_meta["docs"][idx], "score": float(tfidf_scores[idx])}
                )

        results = sorted(results, key=lambda r: -r["score"])[: self.k]
        state["retrieved"] = results
        return state


# Generator Node
class GeneratorNode:
    def __init__(self, prompt_path="prompts-version-1.txt", model="mistral-medium"):
        self.prompt_path = prompt_path
        # Mistral API key must be set in env var
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("Please set MISTRAL_API_KEY in your environment.")
        self.llm = ChatMistralAI(model=model, api_key=api_key, temperature=0.0)

    def _build_prompt(self, question, retrieved_docs):
        context = "\n\n---\n\n".join(
            [f"[{d['doc_id']}]\n{d['text'][:2000]}" for d in retrieved_docs]
        )
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            prompt_txt = f.read()

        instruction = ""
        if "# instruction:" in prompt_txt:
            instruction = prompt_txt.split("# instruction:")[1].split("#")[0].strip()

        system = instruction
        user = f"Context:\n{context}\n\nQuestion:\n{question}\n\nFormatting rules: Use JSON as specified."
        return system, user

    def __call__(self, state):
        q = state["question"]
        retrieved = state["retrieved"]
        system, user = self._build_prompt(q, retrieved)

        resp = self.llm.invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )

        text = resp.content
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = {"answer": text.strip(), "sources": [], "confidence": "low"}

        state["generated"] = parsed
        return state

# Chatbot Graph Wrapper
class ChatbotGraph:
    def __init__(self, indices_dir="indices"):
        router = RouterNode()
        retriever = RetrieverNode(indices_dir=indices_dir, k=20)
        generator = GeneratorNode()

        graph = StateGraph(ChatbotState)
        graph.add_node("router", router)
        graph.add_node("retriever", retriever)
        graph.add_node("generator", generator)

        graph.set_entry_point("router")
        graph.add_edge("router", "retriever")
        graph.add_edge("retriever", "generator")
        graph.add_edge("generator", END)

        self.app = graph.compile()

    def respond(self, question: str) -> Dict[str, Any]:
        state = ChatbotState(question=question)
        out = self.app.invoke(state)
        return {
            "category": out["category"],
            "retrieved": out["retrieved"],
            "generated": out["generated"],
        }


if __name__ == "__main__":
    import sys

    cb = ChatbotGraph()
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Question: ")
    out = cb.respond(q)
    print(json.dumps(out, indent=2, ensure_ascii=False))