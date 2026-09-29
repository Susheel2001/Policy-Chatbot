from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any

from .llm_client import OpenAICompatibleClient
from .models import AgentResult, RetrievedChunk
from .retrieval import Retriever, excerpt_for_chunk, tokenize


class PolicySupportAgent:
    def __init__(
        self,
        retriever: Retriever,
        llm_client: OpenAICompatibleClient | None = None,
        retrieval_threshold: float = 0.65,
    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client
        self.retrieval_threshold = retrieval_threshold

    def answer(self, query: str) -> AgentResult:
        query = (query or "").strip()
        if not query:
            return self._escalate(query, "Empty question")

        category = categorize_query(query)
        retrieved = self.retriever.search(query, top_k=4)
        if not retrieved:
            return self._escalate(query, "No relevant policy text was found")

        top_score = retrieved[0].score
        if top_score < self.retrieval_threshold:
            return self._escalate(query, "Retrieved evidence is too weak to answer safely", retrieved)

        if self.llm_client and self.llm_client.enabled:
            try:
                response = self._answer_with_llm(query, category, retrieved)
                if response.action == "respond" and response.answer:
                    response.sources = self._serialize_sources(query, retrieved)
                    response.category = category
                    return response
                if response.action == "escalate":
                    response.sources = self._serialize_sources(query, retrieved)
                    response.category = "escalation"
                    if not response.reason:
                        response.reason = "Model requested escalation"
                    return response
            except Exception as exc:
                fallback = self._extractive_answer(query, category, retrieved)
                fallback.reason = f"LLM fallback used after error: {exc}"
                fallback.sources = self._serialize_sources(query, retrieved)
                return fallback

        fallback = self._extractive_answer(query, category, retrieved)
        fallback.sources = self._serialize_sources(query, retrieved)
        return fallback

    def _answer_with_llm(
        self,
        query: str,
        category: str,
        retrieved: list[RetrievedChunk],
    ) -> AgentResult:
        context_lines = []
        for i, item in enumerate(retrieved, start=1):
            context_lines.append(
                f"[{i}] doc={item.chunk.doc_name} chunk={item.chunk.chunk_id} score={item.score:.3f}\n"
                f"{excerpt_for_chunk(item.chunk.text, query)}"
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a policy-aware customer support assistant. "
                    "Answer only from the provided policy excerpts. "
                    "Return a single JSON object and nothing else. "
                    "If the excerpts do not support a safe answer, set action to escalate, answer to null, "
                    "confidence to low, and provide a short reason."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\n"
                    f"Detected category: {category}\n\n"
                    "Policy excerpts:\n"
                    + "\n\n".join(context_lines)
                    + "\n\n"
                    "Required JSON keys: query, category, answer, confidence, action, sources, reason.\n"
                    "Allowed confidence values: low, medium, high.\n"
                    "Allowed action values: respond, escalate.\n"
                    "In sources, include the doc name and chunk id used to justify the answer."
                ),
            },
        ]

        raw = self.llm_client.chat(messages=messages, temperature=0.1, max_tokens=700)
        parsed = parse_json_object(raw)
        response = validate_response_dict(parsed, query=query, category=category)
        response.raw_model_output = raw
        return response

    def _extractive_answer(
        self,
        query: str,
        category: str,
        retrieved: list[RetrievedChunk],
    ) -> AgentResult:
        top = retrieved[0]
        sentence = best_sentence(top.chunk.text, query)
        if not sentence or _looks_noisy(sentence) or len(sentence) < 35:
            sentence = excerpt_for_chunk(top.chunk.text, query)
        if not sentence or _looks_noisy(sentence):
            sentence = f"Relevant policy text was found in {top.chunk.doc_name}. Review the evidence panel for the supporting excerpt."

        return AgentResult(
            query=query,
            category=category,
            answer=sentence,
            confidence="medium" if top.score >= self.retrieval_threshold * 1.5 else "low",
            action="respond" if top.score >= self.retrieval_threshold else "escalate",
            reason=None if top.score >= self.retrieval_threshold else "Insufficient evidence to answer confidently",
            sources=[],
        )

    def _escalate(
        self,
        query: str,
        reason: str,
        retrieved: list[RetrievedChunk] | None = None,
    ) -> AgentResult:
        return AgentResult(
            query=query,
            category="escalation",
            answer=None,
            confidence="low",
            action="escalate",
            reason=reason,
            sources=self._serialize_sources(query, retrieved or []),
        )

    @staticmethod
    def _serialize_sources(query: str, retrieved: list[RetrievedChunk]) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        for item in retrieved[:3]:
            sources.append(
                {
                    "doc": item.chunk.doc_name,
                    "chunk_id": item.chunk.chunk_id,
                    "score": round(item.score, 4),
                    "excerpt": excerpt_for_chunk(item.chunk.text, query),
                }
            )
        return sources


def categorize_query(query: str) -> str:
    q = query.lower()
    if any(term in q for term in ["claim", "cashless", "reimburse", "reimbursement", "settlement"]):
        return "claims"
    if any(term in q for term in ["renew", "renewal", "grace period", "premium", "installment"]):
        return "renewal"
    if any(term in q for term in ["cover", "coverage", "benefit", "sum insured", "hospital", "day care"]):
        return "coverage"
    if any(term in q for term in ["pre-existing", "pre existing", "definition", "means", "what is"]):
        return "definitions"
    return "policy_info"


def parse_json_object(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def validate_response_dict(payload: dict[str, Any], query: str, category: str) -> AgentResult:
    action = str(payload.get("action", "respond")).lower()
    if action not in {"respond", "escalate"}:
        action = "respond"

    confidence = str(payload.get("confidence", "medium")).lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"

    answer = payload.get("answer")
    if answer is not None:
        answer = str(answer).strip() or None

    reason = payload.get("reason")
    if reason is not None:
        reason = str(reason).strip() or None

    sources: list[dict[str, Any]] = []
    raw_sources = payload.get("sources") or []
    if isinstance(raw_sources, list):
        for item in raw_sources:
            if isinstance(item, dict):
                sources.append(item)

    parsed_category = str(payload.get("category", category) or category)
    parsed_query = str(payload.get("query", query) or query)
    return AgentResult(
        query=parsed_query,
        category=parsed_category,
        answer=answer,
        confidence=confidence,
        action=action,
        sources=sources,
        reason=reason,
    )


def best_sentence(text: str, query: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip())
    if not sentences:
        return ""
    q_tokens = set(tokenize(query))
    best = ""
    best_score = -1
    for sentence in sentences:
        tokens = set(tokenize(sentence))
        score = len(tokens & q_tokens)
        if score > best_score and sentence:
            best_score = score
            best = sentence
    return best if best_score > 0 else sentences[0]
def _looks_noisy(text: str) -> bool:
    if not text:
        return True
    letters = sum(1 for char in text if char.isalpha())
    if letters < 8:
        return True
    noise_chars = sum(1 for char in text if not (char.isalnum() or char.isspace() or char in ".,;:!?()-/'"))
    return noise_chars / max(1, len(text)) > 0.12

