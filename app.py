from __future__ import annotations

import json
from pathlib import Path

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover - only used when streamlit is missing
    raise SystemExit(
        "Streamlit is not installed. Run `pip install -r requirements.txt` before starting the app."
    ) from exc

from src.agent import PolicySupportAgent
from src.llm_client import load_llm_client_from_env
from src.retrieval import Retriever


ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = ROOT / "data" / "documents"


@st.cache_resource(show_spinner=False)
def build_agent() -> PolicySupportAgent:
    retriever = Retriever.from_document_dir(DOCUMENTS_DIR)
    llm_client = load_llm_client_from_env()
    if not llm_client.enabled:
        llm_client = None
    return PolicySupportAgent(retriever=retriever, llm_client=llm_client)


def load_sample_questions() -> list[dict[str, str]]:
    sample_path = ROOT / "data" / "sample_questions.json"
    if not sample_path.exists():
        return []
    return json.loads(sample_path.read_text(encoding="utf-8"))


st.set_page_config(page_title="Policy Coverage Chatbot", page_icon="??", layout="wide")
st.title("Policy Coverage Chatbot")
st.caption("A small policy-aware customer support prototype for the AI Engineer screening assignment.")

try:
    agent = build_agent()
except ValueError as exc:
    st.error(f"Document loading failed: {exc}")
    st.info("Add at least one supported document to data/documents, then restart the app.")
    st.stop()
samples = load_sample_questions()

with st.sidebar:
    st.header("Setup")
    llm_enabled = bool(agent.llm_client and agent.llm_client.enabled)
    st.write(f"LLM enabled: {'yes' if llm_enabled else 'no'}")
    if llm_enabled and agent.llm_client:
        st.write(f"Model: {agent.llm_client.config.model}")
        st.write(f"Base URL: {agent.llm_client.config.base_url}")
    else:
        st.write("Running in grounded extractive fallback mode.")
    st.write("Sources loaded from `data/documents` (including subfolders).")
    if samples:
        st.write("Sample questions are available below the input.")

question = st.text_area(
    "Ask a policy question",
    placeholder="Example: Are pre-hospitalisation expenses covered?",
    height=120,
)

col1, col2 = st.columns([1, 2])
with col1:
    submit = st.button("Get answer", type="primary", use_container_width=True)
with col2:
    if samples:
        sample_choice = st.selectbox(
            "Load a sample question",
            options=[""] + [item["question"] for item in samples],
            index=0,
        )
        if sample_choice and not question.strip():
            question = sample_choice

if submit:
    result = agent.answer(question)
    st.subheader("Result")
    st.json(result.to_dict())

    if result.sources:
        st.subheader("Evidence")
        for source in result.sources:
            with st.expander(f"{source.get('doc')} - {source.get('chunk_id')}"):
                st.write(f"Score: {source.get('score')}")
                st.write(source.get("excerpt"))
    else:
        st.info("No supporting sources were attached to this response.")

st.divider()
st.subheader("Sample questions")
if samples:
    for item in samples:
        st.write(f"- {item['question']}")
else:
    st.write("No sample questions loaded.")
