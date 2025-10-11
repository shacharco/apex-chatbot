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
import httpx
import joblib
import numpy as np
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from tenacity import RetryError


class ChatbotState(TypedDict):
    question: str
    categories: str
    retrieved: List[str]
    generated: str
    iteration_count: int
    question_history: List[str]  # Track all rephrased questions
    last_context: str  # Store the last retrieved context
    quality_check: Dict[str, Any]  # Store quality check results


init_state = ChatbotState(
    question="",
    categories="",
    retrieved=[],
    generated="",
    iteration_count=0,
    question_history=[],
    last_context="",
    quality_check={}
)

# Router Node
class RouterNode:
    def __init__(self, categories=None, prompt_path="prompts/router/v0.txt", model="mistral-medium", max_retries=5):
        # api_key = os.environ.get("MISTRAL_API_KEY")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set OPENAI_API_KEY in your environment.")
        self.categories = categories or ["car", "life", "travel", "health", "business", "apartment"]
        self.prompt_path = prompt_path
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            self.prompt_txt = f.read()
        self.max_retries = max_retries
        # self.llm = ChatMistralAI(model=model, api_key=api_key, temperature=0.0)
        self.llm = ChatOpenAI(model=model, api_key=api_key)

        self.response_schemas = [
            ResponseSchema(
                name="routes",
                description=(
                    f"A list of all categories that are relevant or possibly relevant to the input text. "
                    f"Each element should be one of the following: {', '.join(self.categories)}. "
                    f"If no categories seem relevant, return an empty list []."
                ),
                type="array[string]"
            ),
            ResponseSchema(name="explanations",
                           description="Explanation of why you chose this answer and explanation why you didnt choose other answers. answer only in one line string format.",
                           type="string")

        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schemas)
        self.kws = {
            "car": [
                "רכב", "תאונה", "חלון", "דלק"
            ],
            "life": [
                "חיים", "בן/בת זוג", "מוטב", "תמותה"
            ],
            "travel": [
                "נסיעה", "טיול", "טיסה", "מטען", "נסיעות", "חו\"ל"
            ],
            "apartment": [
                "דירה", "שוכר", "משכיר", "שריפה"
            ],
            "business": [
                "עסק", "עובד", "חברה", "מפעל", "משרד"
            ],
            "health": [
                "בריאות", "רופא", "חולים", "תרופות", "ניתוח"
            ]
        }

    def __call__(self, state):
        q = state["question"].lower()
        format_instruction = self.output_parser.get_format_instructions()
        messages = [("system", self.prompt_txt), ("user", "{user_prompt}")]

        prompt_template = ChatPromptTemplate.from_messages(
            messages
        )
        prompt = prompt_template.format_prompt(user_prompt=q, format_instructions=format_instruction,
                                               categories=', '.join(self.categories), categories_keywords=self.kws)
        for attempt in range(1, self.max_retries + 1):
            try:
                output = self.llm.invoke(prompt)
                parsed = self.output_parser.parse(output.content)
                best_categories = parsed['routes']
                state["categories"] = best_categories
                for cat in best_categories:
                    if cat not in self.categories:
                        raise ValueError(f"Invalid category '{cat}' returned by the model.")
                break  # success, exit the retry loop
            except (ConnectionError, TimeoutError, ValueError, httpx.HTTPStatusError) as e:
                print(f"Attempt {attempt} failed: {e}")
                if attempt == self.max_retries:
                    raise  # re-raise the exception after max retries
                time.sleep(attempt)
        return state

class RetrieverNode:
    def __init__(self, indices_dir="indices", k=5, alpha=0.5, max_retries=5):
        """
        alpha = weight of TF-IDF score
        (1-alpha) = weight of semantic similarity
        """
        self.indices_dir = indices_dir
        self.k = k
        self.alpha = alpha
        self.max_retries = max_retries
        self.loaded = {}
        self.embedder = MistralAIEmbeddings(model="mistral-embed")
        self.categories = ["car", "life", "travel", "health", "business", "apartment"]


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
        categories = state.get("categories", [])
        question = state.get("question")
        if not question:
            state["retrieved"] = []
            return state
        if len(categories) == 0:
            categories = self.categories
        all_results = []

        # Compute query embedding once
        for attempt in range(1, self.max_retries + 1):
            try:
                q_vec = np.array(self.embedder.embed_query(question)).astype("float32").reshape(1, -1)
                q_norm = np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-9
                qn = q_vec / q_norm
                break
            except RetryError as e:
                print(f"Attempt {attempt} failed: {e}")
                if attempt == self.max_retries:
                    raise  # re-raise the exception after max retries
                time.sleep(attempt)

        for category in categories:
            data = self._load_category(category)
            vec, docs_meta, index = data["vec"], data["docs_meta"], data["index"]
            docs = docs_meta["docs"]

            # --- TF-IDF score ---
            tfidf_scores = vec.transform([question]).dot(vec.transform(docs).T).toarray()[0]
            if np.max(tfidf_scores) > 0:
                tfidf_scores = tfidf_scores / (np.max(tfidf_scores) + 1e-9)
            else:
                tfidf_scores = np.zeros_like(tfidf_scores)

            # --- Semantic embedding score ---
            D, I = index.search(qn, self.k)
            sem_scores = np.zeros(len(docs), dtype=np.float32)
            for score, idx in zip(D[0], I[0]):
                sem_scores[idx] = score
            if np.max(sem_scores) > 0:
                sem_scores = sem_scores / (np.max(sem_scores) + 1e-9)

            # --- Hybrid score ---
            final_scores = self.alpha * tfidf_scores + (1 - self.alpha) * sem_scores

            # --- Collect top results for this category ---
            top_idx = np.argsort(-final_scores)[:self.k]
            for idx in top_idx:
                all_results.append({
                    "category": category,
                    "doc_id": docs_meta["doc_ids"][idx],
                    "text": docs[idx],
                    "title": docs_meta["titles"][idx],
                    "score": float(final_scores[idx])
                })

        # --- Global top-k across all categories ---
        all_results = sorted(all_results, key=lambda x: -x["score"])[:self.k]

        state["retrieved"] = all_results

        # Store last context for quality checking
        context = "\n\n---\n\n".join(
            [f"[{d['doc_id']}]\n{d['text']}" for d in all_results]
        )
        state["last_context"] = context

        return state


# Generator Node
class GeneratorNode:
    def __init__(self, prompt_path="prompts/generator/v0.txt", model="mistral-medium", max_retries=3):
        self.prompt_path = prompt_path
        # api_key = os.environ.get("MISTRAL_API_KEY")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set OPENAI_API_KEY in your environment.")
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            self.prompt_txt = f.read()
        self.max_retries = max_retries
        # self.llm = ChatMistralAI(model=model, api_key=api_key, temperature=0.0)
        self.llm = ChatOpenAI(model=model, api_key=api_key, temperature=0.0)

        # Define schema for JSON output
        self.response_schemas = [
            ResponseSchema(name="answer", description="The main short final answer to the user's question. yes/no/numeric value with units, no full sentence"),
            ResponseSchema(name="sources", description="List of document IDs used for the answer", type="list"),
            ResponseSchema(name="explanations", description="Explanation of why you chose this answer and explanation why you didnt choose other answers")
        ]

        self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schemas)

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        q = state["question"]
        retrieved = state["retrieved"]
        question_history = state.get("question_history", [])
        last_context = state.get("last_context", "")

        # Use last_context if available (from retriever), otherwise build from retrieved
        if not last_context:
            context = "\n\n---\n\n".join(
                [f"[{d['doc_id']}]\n{d['text']}" for d in retrieved]
            )
        else:
            context = last_context

        format_instruction = self.output_parser.get_format_instructions()
        messages = [("system", self.prompt_txt)]

        # Build the user prompt with question history
        user_prompt = f"## Original User Question:\n{question_history[0] if question_history else q}\n\n"
        if len(question_history) > 0 and question_history[-1] != q:
            user_prompt += f"## Previous Search Attempts:\n"
            for i, prev_q in enumerate(question_history):
                user_prompt += f"{i+1}. {prev_q}\n"
            user_prompt += f"\n## Current Search Question:\n{q}\n\n"

        user_prompt += f"## Retrieved Context:\n{context}\n\n## Your Task:\nAnswer the original user question based on the context."

        messages.append(("user", user_prompt))
        messages.append(("user", "{format_instructions}"))

        prompt_template = ChatPromptTemplate.from_messages(
            messages
        )
        prompt = prompt_template.format_prompt(format_instructions=format_instruction)
        for attempt in range(1, self.max_retries + 1):
            try:
                output = self.llm.invoke(prompt)
                parsed = self.output_parser.parse(output.content)
                state["generated"] = parsed
                break  # success, exit the retry loop
            except (ConnectionError, TimeoutError, ValueError, httpx.HTTPStatusError) as e:
                print(f"Attempt {attempt} failed: {e}")
                if attempt == self.max_retries:
                    raise  # re-raise the exception after max retries
                time.sleep(1)
        # Store into state
        return state


# Quality Checker Node
class QualityCheckerNode:
    def __init__(self, prompt_path="prompts/quality_checker/v0.txt", model="mistral-medium", max_retries=3):
        self.prompt_path = prompt_path
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set OPENAI_API_KEY in your environment.")
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            self.prompt_txt = f.read()
        self.max_retries = max_retries
        self.llm = ChatOpenAI(model=model, api_key=api_key, temperature=0.0)
        self.categories = ["car", "life", "travel", "health", "business", "apartment"]

        # Define schema for JSON output
        self.response_schemas = [
            ResponseSchema(
                name="quality",
                description="Either 'good' if the answer is satisfactory, or 'bad' if it needs improvement",
                type="string"
            ),
            ResponseSchema(
                name="suggested_question",
                description="If quality is 'bad', suggest a rephrased question to retrieve better context. Otherwise leave empty.",
                type="string"
            ),
            ResponseSchema(
                name="suggested_categories",
                description=f"If quality is 'bad', suggest different categories to search (as a list). Otherwise leave empty."
                            f"Each element should be one of the following: {', '.join(self.categories)}. "
                    f"If no categories seem relevant, return an empty list [].",
                type="list"
            ),
            ResponseSchema(
                name="explanation",
                description="Brief explanation of the quality assessment",
                type="string"
            )
        ]

        self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schemas)

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        current_question = state["question"]
        question_history = state.get("question_history", [])

        # Determine the original question
        original_question = question_history[0] if question_history else current_question

        generated = state["generated"]
        answer = generated.get("answer", "")
        sources = generated.get("sources", [])
        retrieved = state.get("retrieved", [])

        # Filter context to only include documents cited in sources
        cited_docs = []
        for doc in retrieved:
            doc_id = doc.get("doc_id", "")
            # Check if this doc_id is in the sources list
            if any(doc_id in str(source) for source in sources):
                cited_docs.append(doc)

        # Build context from only the cited documents
        if cited_docs:
            filtered_context = "\n\n---\n\n".join(
                [f"[{d['doc_id']}]\n{d['text']}" for d in cited_docs]
            )
        else:
            # Fallback to empty if no sources were cited
            filtered_context = "No documents were cited in the answer."

        format_instruction = self.output_parser.get_format_instructions()

        messages = [("system", self.prompt_txt)]
        user_message = f"""## Original User Question:
{original_question}

## Current Search Question:
{current_question}

## Retrieved Context (only documents cited in the answer):
{filtered_context}

## Generated Answer:
{answer}

## Your Task:
Evaluate if this answer is good enough to answer the ORIGINAL question, or if we need to search again with a different approach.
"""
        messages.append(("user", user_message))
        messages.append(("user", "{format_instructions}"))

        prompt_template = ChatPromptTemplate.from_messages(messages)
        prompt = prompt_template.format_prompt(format_instructions=format_instruction)

        for attempt in range(1, self.max_retries + 1):
            try:
                output = self.llm.invoke(prompt)
                parsed = self.output_parser.parse(output.content)

                # Store quality check results in state
                state["quality_check"] = parsed

                # If quality is bad and we haven't exceeded iteration limit, update for retry
                if parsed["quality"] == "bad" and state["iteration_count"] < 3:
                    # Store current question in history if not already there
                    if not question_history or question_history[-1] != current_question:
                        # Make sure we store the original question first if this is the first iteration
                        if not question_history:
                            state["question_history"].append(current_question)

                    # Update question if suggested
                    if parsed.get("suggested_question"):
                        state["question"] = parsed["suggested_question"]

                    # Update categories if suggested
                    if parsed.get("suggested_categories") and len(parsed["suggested_categories"]) > 0:
                        state["categories"] = parsed["suggested_categories"]

                break  # success, exit the retry loop
            except (ConnectionError, TimeoutError, ValueError, httpx.HTTPStatusError) as e:
                print(f"Attempt {attempt} failed: {e}")
                if attempt == self.max_retries:
                    raise  # re-raise the exception after max retries
                time.sleep(1)

        return state


# Chatbot Graph Wrapper
class ChatbotGraph:
    def __init__(self, indices_dir="indices", model="mistral-medium"):
        router = RouterNode(model=model)
        retriever = RetrieverNode(indices_dir=indices_dir, k=20, alpha=0.5)
        generator = GeneratorNode(model=model)
        quality_checker = QualityCheckerNode(model=model)

        graph = StateGraph(ChatbotState)
        graph.add_node("router", router)
        graph.add_node("retriever", retriever)
        graph.add_node("generator", generator)
        graph.add_node("quality_checker", quality_checker)

        # Define the routing logic after quality check
        def should_continue(state: ChatbotState) -> str:
            """Determine if we should loop back or end"""
            iteration_count = state.get("iteration_count", 0)
            quality_check = state.get("quality_check", {})

            # If we've reached max iterations (3), end
            if iteration_count >= 3:
                return "end"

            # If quality is good, end
            if quality_check.get("quality") == "good":
                return "end"

            # Otherwise, loop back to retriever (quality is bad and we have iterations left)
            return "continue"

        # Define iteration counter node
        def increment_iteration(state: ChatbotState) -> ChatbotState:
            """Increment the iteration counter"""
            state["iteration_count"] = state.get("iteration_count", 0) + 1
            return state

        graph.add_node("increment_iteration", increment_iteration)

        # Set up the graph flow
        graph.set_entry_point("router")
        graph.add_edge("router", "retriever")
        graph.add_edge("retriever", "generator")
        graph.add_edge("generator", "quality_checker")

        # Conditional edge from quality_checker
        graph.add_conditional_edges(
            "quality_checker",
            should_continue,
            {
                "continue": "increment_iteration",
                "end": END
            }
        )

        # Loop back from increment_iteration to retriever
        graph.add_edge("increment_iteration", "retriever")

        self.app = graph.compile()

    def respond(self, question: str) -> Dict[str, Any]:
        state = ChatbotState(
            question=question,
            categories="",
            retrieved=[],
            generated="",
            iteration_count=0,
            question_history=[],
            last_context="",
            quality_check={}
        )
        out = self.app.invoke(state)
        return {
            "categories": out["categories"],
            "retrieved": out["retrieved"],
            "generated": out["generated"],
            "iteration_count": out.get("iteration_count", 0),
            "quality_check": out.get("quality_check", {}),
        }


if __name__ == "__main__":
    import sys

    cb = ChatbotGraph()
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Question: ")
    out = cb.respond(q)
    print(json.dumps(out, indent=2, ensure_ascii=False))