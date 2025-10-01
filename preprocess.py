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


def tokenize_words(text):
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def chunk_paragraphs(text, target_tokens=300, min_tokens=60, separator="\n\n"):
    """Split into chunks using a separator, then re-chunk based on token counts."""
    paras = [p.strip() for p in text.split(separator) if p.strip()]
    chunks, buffer = [], None
    for p in paras:
        words = tokenize_words(p)
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
    docs, doc_ids = [], [], []
    for path in Path(category_dir).rglob("*"):
        if path.suffix.lower() in [".txt", ".pdf", ".aspx", ".html", ".htm"]:
            chunks = load_and_split(path, target_chunk_tokens, min_paragraph_tokens)
            for i, (c,t) in enumerate(chunks):
                doc_id = f"{path.stem}::p{i}"
                doc_with_title = f"{t}\n\n{c}"
                docs.append(doc_with_title)
                doc_ids.append(doc_id)

    if not docs:
        print(f"No docs found in {category_dir}")
        return

    # TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
    X = vectorizer.fit_transform(docs)

    # FAISS
    X_dense = X.toarray().astype("float32")
    dim = X_dense.shape[1]
    index = faiss.IndexFlatIP(dim)
    norms = np.linalg.norm(X_dense, axis=1, keepdims=True) + 1e-9
    Xn = X_dense / norms
    index.add(Xn)

    # Save
    os.makedirs(out_dir, exist_ok=True)
    base = Path(out_dir) / category_dir.name
    base.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(base / "faiss.index"))
    with open(base / "tfidf_vectorizer.joblib", "wb") as f:
        dump(vectorizer, f)
    with open(base / "doc_ids.pkl", "wb") as f:
        pickle.dump({"doc_ids": doc_ids, "docs": docs}, f)
    print(f"Saved indices for {category_dir.name} to {base}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Input directory containing category subfolders")
    parser.add_argument("--out_dir", default="indices", help="Output indices directory")
    parser.add_argument("--target_chunk_tokens", type=int, default=300)
    parser.add_argument("--min_paragraph_tokens", type=int, default=60)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    for category_path in input_dir.iterdir():
        if category_path.is_dir():
            build_indices_for_category(category_path, args.out_dir, args.target_chunk_tokens, args.min_paragraph_tokens)
