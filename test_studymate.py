"""
StudyMate AI test suite. Run with:  pytest -v

The tests run in a temporary folder, so they never touch your real
chroma_db index, and they don't call the Gemini API (no key needed).
"""
import os
from pathlib import Path

import pytest

from ingest import index_pdf_files, get_collection
from rag import retrieve_relevant_chunks, answer_question, parse_mcqs

SAMPLE_PDF = Path(__file__).parent / "data" / "sample_biology_notes.pdf"


@pytest.fixture
def indexed_notes(tmp_path, monkeypatch):
    """Indexes the sample PDF into a throwaway ChromaDB inside tmp_path."""
    monkeypatch.chdir(tmp_path)  # the app uses the relative "chroma_db" folder
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    pdf_bytes = SAMPLE_PDF.read_bytes()
    result = index_pdf_files([("sample_biology_notes.pdf", pdf_bytes)])
    return result, pdf_bytes


def test_ingestion(indexed_notes):
    result, _ = indexed_notes
    assert result["status"] == "success"
    assert result["chunks_indexed"] > 0
    assert result["pages_indexed"] == 3


def test_reindexing_same_pdf_does_not_duplicate(indexed_notes):
    result, pdf_bytes = indexed_notes
    first_count = get_collection().count()
    index_pdf_files([("sample_biology_notes.pdf", pdf_bytes)])
    index_pdf_files([("sample_biology_notes.pdf", pdf_bytes), ("sample_biology_notes.pdf", pdf_bytes)])
    assert get_collection().count() == first_count == result["chunks_indexed"]


def test_retrieval_page_1_mitochondria(indexed_notes):
    top = retrieve_relevant_chunks("What is the function of mitochondria and cristae?", top_k=4)[0]
    assert top["doc_name"] == "sample_biology_notes.pdf"
    assert top["page_num"] == 1
    assert "mitochondria" in top["text"].lower() or "atp" in top["text"].lower()


def test_retrieval_page_3_dna(indexed_notes):
    top = retrieve_relevant_chunks("What is the role of DNA Helicase and DNA Ligase in replication?", top_k=2)[0]
    assert top["page_num"] == 3
    assert "helicase" in top["text"].lower() or "ligase" in top["text"].lower()


def test_answer_returns_page_2_source(indexed_notes):
    res = answer_question("Explain the role of RuBisCO in the Calvin cycle")
    assert any(s["page_num"] == 2 for s in res["sources"])


def test_abstains_on_out_of_scope_question(indexed_notes):
    res = answer_question("What is the capital of Australia and who is the prime minister?")
    assert "do not contain" in res["answer"].lower()
    assert res["sources"] == []


def test_parse_mcqs_handles_fences_and_drops_bad_items():
    raw = """```json
    [
      {"question": "Q1?", "options": ["A) a", "B) b", "C) c", "D) d"], "answer": "b) b"},
      {"question": "Missing options", "answer": "A"},
      {"question": "Bad answer", "options": ["A) a", "B) b", "C) c", "D) d"], "answer": "E"}
    ]
    ```"""
    mcqs = parse_mcqs(raw)
    assert len(mcqs) == 1
    assert mcqs[0]["answer"] == "B"


def test_parse_mcqs_rejects_empty_reply():
    with pytest.raises(ValueError):
        parse_mcqs("[]")
