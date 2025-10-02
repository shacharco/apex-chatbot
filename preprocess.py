"""
Preprocess docs (txt, pdf, aspx/html) into per-category paragraph store,
TF-IDF matrix, and FAISS index.

Usage:
    python preprocess.py --input_dir docs --out_dir indices
"""

import os
import argparse
import re
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from tqdm import tqdm
import faiss
from joblib import dump

# New deps
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from langchain_mistralai import MistralAIEmbeddings

def tokenize_words(text):
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def chunk_paragraphs(text, target_tokens=300, min_tokens=60, separator="\n\n"):
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
        if merged and len(tokenize_words(merged[-1])) < min_tokens and len(tokenize_words(c)) < target_tokens * 0.6:
            merged[-1] = merged[-1] + separator + c
        else:
            merged.append(c)
    return merged


def read_txt(path):
    text = Path(path).read_text(encoding="utf-8")
    title = path.stem
    return text, title


def read_pdf(path):
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            texts.append(text.strip())
    title = None
    if reader.metadata and reader.metadata.title:
        title = str(reader.metadata.title)
    if not title:
        title = path.stem
    return "\n\n".join(texts), title


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
    ext = path.suffix.lower()
    if ext == ".txt":
        raw, title = read_txt(path)
        separator = "\n\n"
    elif ext == ".pdf":
        raw, title = read_pdf(path)
        # PDFs usually don’t preserve \n\n well; split by single newline instead
        separator = "\n"
    elif ext in [".aspx", ".html", ".htm"]:
        raw, title = read_aspx(path)
        # HTML paragraphs → treat as block splits
        separator = "\n\n"
    else:
        return []

    chunks = chunk_paragraphs(raw, target_tokens=target_chunk_tokens,
                            min_tokens=min_paragraph_tokens, separator=separator)
    return chunks, [title] * len(chunks)



def build_indices_for_category(category_dir, out_dir, target_chunk_tokens=300, min_paragraph_tokens=60):
    docs, doc_ids, titles = [], [], []

    for path in Path(category_dir).rglob("*"):
        if path.suffix.lower() in [".txt", ".pdf", ".aspx", ".html", ".htm"]:
            chunks, chunk_titles = load_and_split(path, target_chunk_tokens, min_paragraph_tokens)
            chunks_lens = [len(tokenize_words(chunk)) for chunk in chunks]
            print(f"chunking {category_dir.name}{path.name} with {len(chunks)} chunks of max size {max(chunks_lens)} and min size {min(chunks_lens)}")
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
    embedder = MistralAIEmbeddings(model="mistral-embed")
    embeddings = embedder.embed_documents(docs)
    embeddings = np.array(embeddings).astype("float32")

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

    input_dir = Path(args.input_dir)
    done_categories = []
    for category_path in input_dir.iterdir():
        if category_path.is_dir() and category_path.name not in done_categories:
            build_indices_for_category(category_path, args.out_dir, args.target_chunk_tokens, args.min_paragraph_tokens)

