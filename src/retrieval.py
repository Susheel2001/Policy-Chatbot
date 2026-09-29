from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .models import Chunk, RetrievedChunk
from .document_text import SUPPORTED_DOCUMENT_EXTENSIONS, extract_document_blocks


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "may",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "this",
    "to",
    "us",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
    "you",
    "your",
}


TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class Corpus:
    chunks: list[Chunk]


class Retriever:
    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self._chunk_tokens: list[list[str]] = [tokenize(chunk.text) for chunk in corpus.chunks]
        self._avg_len = sum(len(tokens) for tokens in self._chunk_tokens) / max(1, len(self._chunk_tokens))
        self._idf = self._build_idf(self._chunk_tokens)

    @classmethod
    def from_document_dir(cls, document_dir: str | Path, exclude: Iterable[str] | None = None) -> "Retriever":
        exclude_names = {name.lower() for name in (exclude or [])}
        chunks: list[Chunk] = []
        root = Path(document_dir)
        for document_path in sorted(path for path in root.rglob("*") if path.is_file()):
            if document_path.name.lower() in exclude_names or document_path.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
                continue
            blocks = extract_document_blocks(document_path)
            for block_index, block in enumerate(blocks):
                parts = split_text(block["text"], max_chars=900, overlap=120)
                for part_index, part in enumerate(parts):
                    chunks.append(
                        Chunk(
                            doc_name=document_path.relative_to(root).as_posix(),
                            chunk_id=f"{document_path.stem}-{block_index}-{part_index}",
                            text=part,
                            block_id=block["block_id"],
                            title=document_path.stem,
                        )
                    )
        if not chunks:
            raise ValueError(f"No supported documents were loaded from {root}.")
        return cls(Corpus(chunks=chunks))

    @classmethod
    def from_pdf_dir(cls, pdf_dir: str | Path, exclude: Iterable[str] | None = None) -> "Retriever":
        """Backward-compatible alias for the former PDF-only loader."""
        return cls.from_document_dir(pdf_dir, exclude=exclude)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        q_counts = Counter(q_tokens)
        scored: list[RetrievedChunk] = []
        for chunk, tokens in zip(self.corpus.chunks, self._chunk_tokens):
            doc_tf = Counter(tokens)
            doc_len = max(1, len(tokens))
            score = 0.0
            for term, qf in q_counts.items():
                tf = doc_tf.get(term, 0)
                if tf == 0:
                    continue
                idf = self._idf.get(term, 0.0)
                k1 = 1.5
                b = 0.75
                denom = tf + k1 * (1 - b + b * doc_len / max(1e-6, self._avg_len))
                score += idf * tf * (k1 + 1) / denom * (1.0 + math.log1p(qf))
            if score > 0:
                scored.append(RetrievedChunk(chunk=chunk, score=score))

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    @staticmethod
    def _build_idf(token_lists: list[list[str]]) -> dict[str, float]:
        df: defaultdict[str, int] = defaultdict(int)
        for tokens in token_lists:
            for term in set(tokens):
                df[term] += 1
        total_docs = max(1, len(token_lists))
        return {term: math.log((total_docs - freq + 0.5) / (freq + 0.5) + 1.0) for term, freq in df.items()}


def tokenize(text: str) -> list[str]:
    tokens = [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS]
    return tokens


def split_text(text: str, max_chars: int = 900, overlap: int = 100) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_chars:
        return [normalized]

    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current.strip())
        if len(sentence) <= max_chars:
            current = sentence
        else:
            step = max(1, max_chars - overlap)
            for start in range(0, len(sentence), step):
                chunks.append(sentence[start : start + max_chars].strip())
            current = ""
    if current:
        chunks.append(current.strip())
    return [chunk for chunk in chunks if chunk]


def excerpt_for_chunk(chunk_text: str, query: str, max_chars: int = 320) -> str:
    normalized = re.sub(r"\s+", " ", chunk_text).strip()
    q_tokens = [token for token in tokenize(query) if len(token) > 2]
    if not q_tokens:
        return normalized[:max_chars]

    best_index = -1
    best_term = ""
    lowered = normalized.lower()
    for term in q_tokens:
        index = lowered.find(term)
        if index != -1 and (best_index == -1 or index < best_index):
            best_index = index
            best_term = term
    if best_index == -1:
        return normalized[:max_chars]

    start = max(0, best_index - max_chars // 3)
    end = min(len(normalized), start + max_chars)
    snippet = normalized[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(normalized):
        snippet = snippet + "..."
    return snippet
