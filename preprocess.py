"""
Preprocess docs (txt, pdf, aspx/html) into per-category paragraph store,
TF-IDF matrix, and FAISS index.

Usage:
    python preprocess.py --input_dir docs --out_dir indices
"""

import os
import time

import pdfplumber
import argparse
import re
import pickle
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from tqdm import tqdm
import faiss
from joblib import dump
from pdfminer.high_level import extract_text

# New deps
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from bidi.algorithm import get_display

def tokenize_words(text):
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def chunk_paragraphs(text, target_tokens=300, min_tokens=150, separator="\n\n"):
    """Split into chunks using separator, then force-break long ones."""
    paras = [p.strip() for p in text.split(separator) if p.strip()]
    chunks, buffer = [], None

    def split_long(p, max_tokens):
        words = tokenize_words(p)
        if len(words) <= max_tokens:
            return [p]
        # split into ~max_tokens sized chunks
        return [
            " ".join(words[i:i+max_tokens])
            for i in range(0, len(words), max_tokens)
        ]

    for p in paras:
        words = tokenize_words(p)

        # 🔹 Hard split if paragraph itself is too long
        if len(words) >= target_tokens * 1.5:
            if buffer:
                chunks.append(buffer)
                buffer = None
            chunks.extend(split_long(p, target_tokens))
            continue

        if len(words) >= target_tokens:
            if buffer:
                chunks.append(buffer)
                buffer = None
            chunks.append(p)
        else:
            if buffer is None:
                buffer = p
            else:
                combined = buffer + separator + p
                combined_words = tokenize_words(combined)
                if len(combined_words) <= target_tokens or len(combined_words) < 2 * min_tokens:
                    buffer = combined
                else:
                    chunks.append(buffer)
                    buffer = p
    if buffer:
        chunks.append(buffer)

    # Merge too-small chunks
    merged = []
    for c in chunks:
        if merged and len(tokenize_words(merged[-1])) < min_tokens and len(tokenize_words(c)) < target_tokens * 1.5:
            merged[-1] = merged[-1] + separator + c
        else:
            merged.append(c)
    return merged


def read_txt(path):
    text = Path(path).read_text(encoding="utf-8")
    title = path.stem
    return text, title


def read_aspx(path):
    html = Path(path).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    # Extract visible text, ignoring scripts/styles
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n\s*\n", "\n\n", text).strip()
    # Normalize spacing
    title = soup.title.string.strip() if soup.title and soup.title.string else path.stem
    return text, title



def load_and_split(path, target_chunk_tokens=300, min_paragraph_tokens=60):
    """Read and split document into chunks depending on its type."""
    ext = path.suffix.lower()

    # TXT
    if ext == ".txt":
        raw, title = read_txt(path)
        chunks = chunk_paragraphs(raw, target_tokens=target_chunk_tokens,
                                  min_tokens=min_paragraph_tokens, separator="\n\n")
        titles = [title] * len(chunks)
        return chunks, titles

    # PDF → one chunk per page
    elif ext == ".pdf":
        chunks, titles = read_pdf(path)
        return chunks, titles

    # ASPX / HTML
    elif ext in [".aspx", ".html", ".htm"]:
        raw, title = read_aspx(path)
        chunks = chunk_paragraphs(raw, target_tokens=target_chunk_tokens,
                                  min_tokens=min_paragraph_tokens, separator="\n\n")
        titles = [title] * len(chunks)
        return chunks, titles

    else:
        return []

from docling.document_converter import DocumentConverter
# from docling.chunking import HybridChunker
# from docling.datamodel.document import DoclingDocument

def parse_keywords_and_summary(llm_output):
    """Parse the LLM output to extract keywords and summary"""
    lines = llm_output.strip().split("\n")
    keywords = ""
    summary = ""

    for i, line in enumerate(lines):
        if line.startswith("KEYWORDS:"):
            keywords = line.replace("KEYWORDS:", "").strip()
        elif line.startswith("SUMMARY:"):
            # Get everything after "SUMMARY:" including multi-line content
            summary = "\n".join(lines[i:]).replace("SUMMARY:", "").strip()
            break

    return keywords, summary

def summarize(history, chunk):
    messages = [("system", prompt_txt), ("user", "history: {history}"), ("user", "current chunk: {chunk}")]
    prompt_template = ChatPromptTemplate.from_messages(
        messages
    )
    prompt = prompt_template.format_prompt(history=history, chunk=chunk)
    for attempt in range(6):
        try:
            output = llm.invoke(prompt)
            return output.content
        except Exception as e:
            print(e)
            if attempt == 5:
                raise
            time.sleep(1)

def read_pdf(path: str, min_chars: int = 200,
                    max_tokens: int = 2000, overlap_tokens: int = 100):
    """
    Read and chunk a PDF with contextual information.

    Each chunk (after the first) will have the following format:
    KEYWORDS: <keywords from previous chunk>

    SUMMARY: <summary of previous chunk>

    <<<CONTENT_START>>>

    <actual chunk content>

    To extract only content in postprocessing, split by '<<<CONTENT_START>>>' and take the last part.
    """
    # Convert
    title = Path(path).stem
    converter = DocumentConverter()
    converted = converter.convert(path)
    doc = converted.document            # a DoclingDocument instance
    markdown = doc.export_to_markdown()
    chunks = chunk_paragraphs(markdown)

    # Define unique separator
    SEPARATOR = "<<<CONTENT_START>>>"

    # Process chunks starting from the second one
    for i in range(1, len(chunks)):
        llm_output = summarize(chunks[i-1], chunks[i])
        # keywords, summary = parse_keywords_and_summary(llm_output)

        # Format: KEYWORDS\nSUMMARY\nSEPARATOR\nCONTENT
        # contextual_prefix = f"KEYWORDS: {keywords}\n\nSUMMARY: {summary}\n\n{SEPARATOR}\n\n"
        contextual_prefix = f"{llm_output}\n\n{SEPARATOR}\n\n"
        chunks[i] = f"{contextual_prefix}{chunks[i]}"

    return chunks, [title for _ in chunks]

def build_indices_for_category(category_dir, out_dir, target_chunk_tokens=300, min_paragraph_tokens=60):
    docs, doc_ids, titles = [], [], []

    for path in Path(category_dir).rglob("*"):
        if path.suffix.lower() in [".txt", ".pdf", ".aspx", ".html", ".htm"]:
            chunks, chunk_titles = load_and_split(path, target_chunk_tokens, min_paragraph_tokens)
            chunks_lens = [len(tokenize_words(chunk)) for chunk in chunks]
            print(f"chunking {category_dir.name} {path.name} with {len(chunks)} chunks of max size {max(chunks_lens)} and min size {min(chunks_lens)}")
            print(chunks_lens)
            for i, (c, t) in enumerate(zip(chunks, chunk_titles)):
                doc_id = f"{path.stem}::p{i}"
                doc_with_title = f"{t}\n\n{c}"  # prepend title for context
                docs.append(doc_with_title)
                doc_ids.append(doc_id)
                titles.append(t)

    if not docs:
        print(f"No docs found in {category_dir}")
        return

    # 1️⃣ TF-IDF4
    print(f"starting TF-IDF for {category_dir.name}")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
    X_tfidf = vectorizer.fit_transform(docs)

    # 2️⃣ Semantic embeddings
    print(f"starting Embeddings for {category_dir.name} with {len(docs)} docs")
    for attempt in range(6):
        try:
            embedder = MistralAIEmbeddings(model="mistral-embed")
            embeddings = embedder.embed_documents(docs)
            embeddings = np.array(embeddings).astype("float32")
            break
        except Exception as e:
            print(e)
            if attempt == 5:
                raise
            time.sleep(1)
    # Normalize semantic embeddings for cosine similarity
    print("normalizing embeddings")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    embeddings_normalized = embeddings / norms

    # FAISS index on semantic embeddings
    print("indexing embeddings")
    dim = embeddings_normalized.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_normalized)

    # Save
    base = Path(out_dir) / category_dir.name
    os.makedirs(base, exist_ok=True)
    faiss.write_index(index, str(base / "faiss.index"))
    dump(vectorizer, base / "tfidf_vectorizer.joblib")
    with open(base / "doc_ids.pkl", "wb") as f:
        pickle.dump({"doc_ids": doc_ids, "docs": docs, "titles": titles}, f)

    print(f"Saved hybrid indices for {category_dir.name} to {base}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Input directory containing category subfolders")
    parser.add_argument("--out_dir", default="indices", help="Output indices directory")
    parser.add_argument("--target_chunk_tokens", type=int, default=300)
    parser.add_argument("--min_paragraph_tokens", type=int, default=60)
    args = parser.parse_args()
    custom_cache = Path(__file__).parent / "hf_cache"
    os.environ["HF_HOME"] = str(custom_cache)
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    model = "mistral-medium"
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("Please set MISTRAL_API_KEY in your environment.")
    llm = ChatMistralAI(model=model, api_key=api_key, temperature=0.0)
    with open("prompts/preprocess/v0.txt", "r", encoding="utf-8") as f:
        prompt_txt = f.read()

    custom_cache.mkdir(exist_ok=True)
    input_dir = Path(args.input_dir)
    done_categories = ["apartment", "car", "business"]
    for category_path in input_dir.iterdir():
        if category_path.is_dir() and category_path.name not in done_categories:
            build_indices_for_category(category_path, args.out_dir, args.target_chunk_tokens, args.min_paragraph_tokens)

