# Policy Coverage Chatbot

A focused Streamlit prototype for the IIFL Finance AI Engineer Round 1 assignment.

## Run locally

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

Optional LLM configuration: copy `.env.example` to `.env`, add a Groq API key, and keep `.env` out of source control. Without a key, the app returns grounded extractive answers.

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=openai/gpt-oss-20b
```

## How the solution works

1. The Streamlit app accepts a customer policy question.
2. It recursively reads documents from `data/documents` and splits extracted text into chunks.
3. A lightweight BM25-style lexical retriever ranks the most relevant chunks.
4. The top evidence is sent to the configured LLM with instructions to answer only from that evidence.
5. The result is validated as structured JSON and includes the answer, confidence, action, and supporting sources.
6. Empty input, missing evidence, weak retrieval, malformed model output, and LLM errors result in escalation or a grounded fallback.

## Why this approach

- Streamlit keeps the interface small and easy to run locally.
- BM25-style retrieval is transparent and sufficient for the small supplied corpus; a vector database is unnecessary here.
- Groq/OpenAI-compatible configuration keeps the model provider replaceable.
- A fixed schema and evidence-first prompt make the LLM response easier to inspect and constrain.

## Documents

Place use-case documents in `data/documents`; subfolders are included and shown in source labels. Supported types are PDF, DOCX, TXT, Markdown, HTML, RTF, CSV, and JSON. Unsupported files are ignored.

## Example

Input:

```text
Are pre-hospitalisation and post-hospitalisation expenses covered?
```

Output shape:

```json
{
  "query": "Are pre-hospitalisation and post-hospitalisation expenses covered?",
  "category": "coverage",
  "answer": "…grounded answer from the retrieved policy text…",
  "confidence": "medium",
  "action": "respond",
  "sources": [{"doc": "4177.pdf", "chunk_id": "4177-…", "score": 1.23, "excerpt": "…"}],
  "reason": null
}
```

`sources` is a list rather than the assignment's singular `source` field so an answer can cite more than one policy excerpt.

## Before production

- Add authentication, authorization, tenant isolation, and secure secret management.
- Use stronger retrieval with embeddings, metadata filters, evaluation sets, and citation quality checks.
- Add safety policies, prompt-injection defenses, audit logs, monitoring, and human-review workflows.
- Validate document upload type, size, malware status, and extraction quality.
- Add automated regression tests and load/error testing around the LLM provider.

## Security and governance concern

Policy and customer queries may contain personal or financial information. Secrets and raw prompts must not be committed, broadly logged, or sent to an unapproved model provider. A production deployment needs data classification, retention controls, access logging, and a reviewed vendor/data-processing posture.

## AI coding tools used

OpenAI Codex was used for code review, debugging, and focused implementation changes. The retrieval logic, output schema, error paths, and test results were reviewed in the repository rather than accepted blindly.

## Tests

```bash
python -m unittest discover -s tests -v
```
