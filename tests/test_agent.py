from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.agent import PolicySupportAgent, parse_json_object
from src.llm_client import LLMConfig, OpenAICompatibleClient, groq_sdk_base_url
from src.pdf_text import extract_pdf_text
from src.retrieval import Retriever


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = ROOT / "data" / "documents"


class DummyLLMClient(OpenAICompatibleClient):
    def __init__(self, content: str):
        super().__init__(LLMConfig(api_key="dummy", base_url="https://example.com/v1", model="dummy"))
        self.content = content

    @property
    def enabled(self) -> bool:
        return True

    def chat(self, messages, temperature=0.1, max_tokens=600):  # noqa: ANN001
        return self.content


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = Retriever.from_document_dir(DOCUMENTS_DIR)

    def test_pdf_loading_creates_chunks(self):
        self.assertGreater(len(self.retriever.corpus.chunks), 10)

    def test_pdf_extraction_contains_real_policy_text(self):
        pdf_path = next(DOCUMENTS_DIR.glob("*.pdf"))
        text = extract_pdf_text(pdf_path)
        lowered = text.lower()
        self.assertIn("day care treatment", lowered)
        self.assertIn("grace period", lowered)

    def test_retrieval_returns_results(self):
        results = self.retriever.search("pre-existing disease coverage", top_k=3)
        self.assertTrue(results)
        self.assertGreater(results[0].score, 0)

    def test_agent_returns_structured_result_without_llm(self):
        agent = PolicySupportAgent(self.retriever, llm_client=None)
        result = agent.answer("Are pre-hospitalisation expenses covered?")
        payload = result.to_dict()
        self.assertIn(payload["action"], {"respond", "escalate"})
        self.assertEqual(payload["query"], "Are pre-hospitalisation expenses covered?")
        self.assertTrue(payload["sources"] or payload["action"] == "escalate")

    def test_all_supplied_sample_questions_return_grounded_results(self):
        agent = PolicySupportAgent(self.retriever, llm_client=None)
        samples = json.loads((ROOT / "data" / "sample_questions.json").read_text(encoding="utf-8"))

        for sample in samples:
            with self.subTest(question=sample["question"]):
                result = agent.answer(sample["question"])
                self.assertEqual(result.action, "respond")
                self.assertTrue(result.answer)
                self.assertTrue(result.sources)

    def test_agent_parses_llm_json(self):
        raw = json.dumps(
            {
                "query": "Are pre-hospitalisation and post-hospitalisation expenses covered?",
                "category": "coverage",
                "answer": "The policy covers pre- and post-hospitalisation expenses.",
                "confidence": "high",
                "action": "respond",
                "sources": [{"doc": "4225.pdf", "chunk_id": "4225-0-0"}],
                "reason": None,
            }
        )
        agent = PolicySupportAgent(self.retriever, llm_client=DummyLLMClient(raw))
        result = agent.answer("Are pre-hospitalisation and post-hospitalisation expenses covered?")
        self.assertEqual(result.action, "respond")
        self.assertEqual(result.confidence, "high")
        self.assertTrue(result.sources)

    def test_empty_query_escalates(self):
        agent = PolicySupportAgent(self.retriever, llm_client=None)
        result = agent.answer("")
        self.assertEqual(result.action, "escalate")
        self.assertEqual(result.confidence, "low")

    def test_parse_json_object_handles_wrapped_json(self):
        wrapped = "Here is the answer: {\"action\": \"escalate\", \"confidence\": \"low\"}"
        parsed = parse_json_object(wrapped)
        self.assertEqual(parsed["action"], "escalate")

    def test_groq_sdk_base_url_uses_api_host(self):
        self.assertEqual(
            groq_sdk_base_url("https://api.groq.com/openai/v1"),
            "https://api.groq.com",
        )


if __name__ == "__main__":
    unittest.main()
