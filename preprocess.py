"""
Preprocess a directory of text documents into per-category paragraph store, TF-IDF matrix and FAISS index.
Usage:
    python preprocess.py --input_dir data/docs --out_dir indices --min_paragraph_tokens 60 --target_chunk_tokens 300

Documents should include category in filename or a metadata file; this script assumes files are arranged per-category subfolder, e.g. data/docs/car_insurance/*.txt
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

# Simple token approximation: use whitespace tokens as proxy for tokens

def tokenize_words(text):
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def chunk_paragraphs(text, target_tokens=300, min_tokens=60):
    # Split naively by double newlines first
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks = []
    buffer = None
    for p in paras:
        words = tokenize_words(p)
        if len(words) >= target_tokens:
            # keep as-is (could be longer than target)
            if buffer:
                # flush buffer
                chunks.append(buffer)
                buffer = None
            chunks.append(p)
        else:
            if buffer is None:
                buffer = p
            else:
                combined = buffer + "\n\n" + p
                combined_words = tokenize_words(combined)
                if len(combined_words) <= target_tokens or len(combined_words) < 2 * min_tokens:
                    buffer = combined
                else:
                    chunks.append(buffer)
                    buffer = p
    if buffer:
        chunks.append(buffer)

    # post-process: merge too small chunks
    merged = []
    for c in chunks:
        if merged and len(tokenize_words(merged[-1])) < min_tokens and len(tokenize_words(c)) < target_tokens * 0.6:
            merged[-1] = merged[-1] + "\n\n" + c
        else:
            merged.append(c)
    return merged


def build_indices_for_category(category_dir, out_dir, target_chunk_tokens=300, min_paragraph_tokens=60):
    docs = []
    doc_ids = []
    for path in Path(category_dir).glob('**/*.txt'):
        text = path.read_text(encoding='utf-8')
        chunks = chunk_paragraphs(text, target_tokens=target_chunk_tokens, min_tokens=min_paragraph_tokens)
        for i, c in enumerate(chunks):
            doc_id = f"{path.stem}::p{i}"
            docs.append(c)
            doc_ids.append(doc_id)
    if not docs:
        print(f"No docs found in {category_dir}")
        return

    # TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=50000)
    X = vectorizer.fit_transform(docs)

    # FAISS: use dense embeddings from TF-IDF as fallback (user can replace with proper embedding model)
    # Convert to float32 dense vectors
    X_dense = X.astype(np.float32).toarray()
    dim = X_dense.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors
    # normalize rows
    norms = np.linalg.norm(X_dense, axis=1, keepdims=True) + 1e-9
    Xn = X_dense / norms
    index.add(Xn)

    os.makedirs(out_dir, exist_ok=True)
    base = Path(out_dir) / category_dir.name
    base.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(base / 'faiss.index'))
    with open(base / 'tfidf_vectorizer.joblib', 'wb') as f:
        dump(vectorizer, f)
    with open(base / 'doc_ids.pkl', 'wb') as f:
        pickle.dump({'doc_ids': doc_ids, 'docs': docs}, f)
    print(f"Saved indices for {category_dir.name} to {base}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Input directory containing category subfolders')
    parser.add_argument('--out_dir', default='indices', help='Output indices directory')
    parser.add_argument('--target_chunk_tokens', type=int, default=300)
    parser.add_argument('--min_paragraph_tokens', type=int, default=60)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    for category_path in input_dir.iterdir():
        if category_path.is_dir():
            build_indices_for_category(category_path, args.out_dir, args.target_chunk_tokens, args.min_paragraph_tokens)

