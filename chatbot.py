# FILE: chatbot.py
"""
LangGraph-based chatbot pipeline.
Nodes: Router -> Retriever -> Generator
"""

import os
import json
import pickle
from typing import List, Dict, Any, TypedDict
import time

import faiss
import joblib
import numpy as np
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

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
        self.categories = categories or ["car", "life", "travel", "health", "business", "apartment"]
        self.kws = {
            "car": [
                "car_insurance", "car", "vehicle", "collision", "comprehensive",
                "auto", "windshield", "license",
                "רכב", "ביטוח רכב", "תאונה", "פגיעה", "רישיון", "חלון קדמי", "קניית רכב"
            ],
            "life": [
                "life", "beneficiary", "death", "term life", "whole life",
                "premium", "insured",
                "חיים", "ביטוח חיים", "בן/בת זוג", "מוטב", "פוליסת חיים", "מנוי", "תמותה"
            ],
            "travel": [
                "travel", "trip", "lost luggage", "cancel", "medical abroad",
                "flight", "delay",
                "נסיעה", "טיול", "ביטול טיסה", "אובדן מטען", "נסיעות", "עזרה רפואית בחו\"ל"
            ],
            "apartment": [
                "apartment", "renters", "tenant", "landlord", "property damage",
                "fire", "theft", "flood",
                "דירה", "שוכר", "משכיר", "נזק לרכוש", "שריפה", "גניבה", "צפה"
            ],
            "business": [
                "business", "commercial", "liability", "worker", "professional",
                "enterprise", "company", "corporate",
                "עסק", "ביטוח עסק", "אחריות מקצועית", "עובד", "חברה", "מפעל", "משרד", "עסקים"
            ],
            "health": [
                "health", "medical", "hospital", "doctor", "treatment",
                "surgery", "insurance card", "prescription", "pharmacy",
                "בריאות", "ביטוח בריאות", "רופא", "בית חולים", "תרופות", "טיפול רפואי", "ניתוח"
            ]
        }

    def __call__(self, state):
        q = state["question"].lower()
        scores = {c: 0 for c in self.categories}
        for c in self.categories:
            for k in self.kws.get(c, []):
                if k in q:
                    scores[c] += 1
        best_category, best_score = max(scores.items(), key=lambda x: x[1])
        if best_score < 0.1:
            best_category = None
        state["category"] = best_category
        return state


class RetrieverNode:
    def __init__(self, indices_dir="indices", k=20, alpha=0.5):
        """
        alpha = weight of TF-IDF score
        (1-alpha) = weight of semantic similarity
        """
        self.indices_dir = indices_dir
        self.k = k
        self.alpha = alpha
        self.loaded = {}
        self.embedder = MistralAIEmbeddings(model="mistral-embed")


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
        category = state.get("category")
        question = state.get("question")
        if not category or not question:
            state["retrieved"] = []
            return state

        data = self._load_category(category)
        vec, docs_meta, index = data["vec"], data["docs_meta"], data["index"]
        docs = docs_meta["docs"]

        # 1️⃣ TF-IDF score
        tfidf_scores = vec.transform([question]).dot(vec.transform(docs).T).toarray()[0]
        tfidf_scores = tfidf_scores / (np.max(tfidf_scores) + 1e-9)  # normalize 0-1

        # 2️⃣ Semantic embeddings score
        q_vec = self.embedder.embed_query(question)
        q_vec = np.array(q_vec).astype("float32").reshape(1, -1)
        q_norm = np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-9
        qn = q_vec / q_norm
        D, I = index.search(qn, self.k)

        sem_scores = np.zeros(len(docs), dtype=np.float32)
        for score, idx in zip(D[0], I[0]):
            sem_scores[idx] = score
        sem_scores = sem_scores / (np.max(sem_scores) + 1e-9)

        # 3️⃣ Hybrid score
        final_scores = self.alpha * tfidf_scores + (1 - self.alpha) * sem_scores

        top_idx = np.argsort(-final_scores)[:self.k]
        results = []
        for idx in top_idx:
            results.append({
                "doc_id": docs_meta["doc_ids"][idx],
                "text": docs[idx],
                "title": docs_meta["titles"][idx],
                "score": float(final_scores[idx])
            })

        state["retrieved"] = results
        return state

# Generator Node
class GeneratorNode:
    def __init__(self, prompt_path="prompts-version-1.txt", model="mistral-medium", max_retries=3):
        self.prompt_path = prompt_path
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("Please set MISTRAL_API_KEY in your environment.")
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            self.prompt_txt = f.read()
        self.max_retries = max_retries
        self.llm = ChatMistralAI(model=model, api_key=api_key, temperature=0.0)

        # Define schema for JSON output
        self.response_schemas = [
            ResponseSchema(name="answer", description="The main answer to the user's question"),
            ResponseSchema(name="sources", description="List of document IDs used for the answer", type="list"),
            ResponseSchema(name="confidence", description="Confidence level in ['low','medium','high']")
        ]

        self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schemas)

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        q = state["question"]
        retrieved = state["retrieved"]
        context = "\n\n---\n\n".join(
            [f"[{d['doc_id']}]\n{d['text'][:2000]}" for d in retrieved]
        )
        format_instruction = self.output_parser.get_format_instructions()
        messages = [("system", self.prompt_txt)]
        context_message = f"""##Context:\n{context}"""
        messages.append(("user", context_message))
        messages.append(("user", "{user_prompt}"))

        prompt_template = ChatPromptTemplate.from_messages(
            messages
        )
        prompt = prompt_template.format_prompt(user_prompt=q, format_instructions=format_instruction)
        for attempt in range(1, self.max_retries + 1):
            try:
                output = self.llm.invoke(prompt)
                parsed = self.output_parser.parse(output.content)
                state["generated"] = parsed
                break  # success, exit the retry loop
            except (ConnectionError, TimeoutError, ValueError) as e:
                print(f"Attempt {attempt} failed: {e}")
                if attempt == self.max_retries:
                    raise  # re-raise the exception after max retries
                time.sleep(1)
        # Store into state
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