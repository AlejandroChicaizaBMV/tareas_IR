from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = PROJECT_ROOT / "nfcorpus" / "corpus.jsonl"

STOPWORDS = set(ENGLISH_STOP_WORDS)
STEMMER = PorterStemmer()
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def tokenize(text: str) -> list[str]:
    terms = TOKEN_RE.findall(str(text).lower())
    return [STEMMER.stem(term) for term in terms if term not in STOPWORDS]


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


if not CORPUS_PATH.exists():
    raise FileNotFoundError(f"No se encuentra el corpus: {CORPUS_PATH}")

docs = [d for d in load_jsonl(CORPUS_PATH) if str(d.get("text", "")).strip()]
doc_ids = [str(d["_id"]) for d in docs]
doc_by_id = {str(d["_id"]): d for d in docs}
doc_tokens = [tokenize(d.get("title", "") + " " + d.get("text", "")) for d in docs]
doc_lengths = np.asarray([len(tokens) for tokens in doc_tokens], dtype=np.float64)
N = len(docs)
avgdl = float(doc_lengths.mean()) if N else 0.0
postings = defaultdict(dict)
for i, terms in enumerate(doc_tokens):
    for term, frequency in Counter(terms).items():
        postings[term][i] = frequency


def identity(tokens):
    return tokens


tfidf = TfidfVectorizer(
    tokenizer=identity,
    preprocessor=identity,
    token_pattern=None,
    lowercase=False,
    norm="l2",
    dtype=np.float32,
)
X_tfidf = tfidf.fit_transform(doc_tokens)


def tfidf_scores(query: str) -> np.ndarray:
    terms = tokenize(query)
    if not terms:
        return np.zeros(N, dtype=np.float64)
    query_vector = tfidf.transform([terms])
    return np.asarray((X_tfidf @ query_vector.T).toarray()).ravel()


def bm25_scores(query: str, k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    scores = np.zeros(N, dtype=np.float64)
    if not avgdl:
        return scores
    for term in set(tokenize(query)):
        posting = postings.get(term)
        if not posting:
            continue
        term_df = len(posting)
        idf = math.log1p((N - term_df + 0.5) / (term_df + 0.5))
        indices = np.fromiter(posting.keys(), dtype=int)
        frequencies = np.fromiter(posting.values(), dtype=np.float64)
        norm = frequencies + k1 * (1 - b + b * doc_lengths[indices] / avgdl)
        scores[indices] += idf * frequencies * (k1 + 1) / norm
    return scores


def rank(model: str, query: str, top_k: int):
    if model == "tfidf":
        scores = tfidf_scores(query)
    elif model == "bm25":
        scores = bm25_scores(query)
    else:
        raise ValueError("model debe ser 'bm25' o 'tfidf'")
    order = np.lexsort((np.asarray(doc_ids), -scores))
    results = []
    for i in order:
        if scores[i] <= 0 or len(results) >= top_k:
            break
        doc = docs[i]
        text = str(doc.get("text", "")).replace("\n", " ")
        results.append({
            "rank": len(results) + 1,
            "document_id": doc_ids[i],
            "score": float(scores[i]),
            "title": doc.get("title") or None,
            "snippet": text[:300],
        })
    return results


app = FastAPI(title="NFCorpus Information Retrieval", version="1.0.0")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    model: str = "bm25"
    top_k: int = Field(default=10, ge=1, le=100)


@app.get("/health")
def health():
    return {"status": "ok", "documents": N, "vocabulary": len(postings)}


@app.post("/search")
def search(request: SearchRequest):
    model = request.model.strip().lower()
    if model not in {"bm25", "tfidf"}:
        raise HTTPException(status_code=400, detail="model debe ser 'bm25' o 'tfidf'")
    return {"model": model, "results": rank(model, request.query, request.top_k)}
