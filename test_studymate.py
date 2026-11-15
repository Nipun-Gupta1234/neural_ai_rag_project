import os
import sys
from ingest import index_pdf_files, clear_knowledge_base, get_collection
from rag import retrieve_relevant_chunks, answer_question

def run_tests():
    print("=" * 60)
    print("🧪 Running StudyMate AI Verification Test Suite")
    print("=" * 60)

    pdf_path = "data/sample_biology_notes.pdf"
    assert os.path.exists(pdf_path), f"Sample PDF not found at {pdf_path}"

    # Step 1: Read PDF bytes and index
    print("\n1. Testing PDF Ingestion and ChromaDB Indexing...")
    clear_knowledge_base()
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    index_result = index_pdf_files([("sample_biology_notes.pdf", pdf_bytes)])
    print(f"   Result: {index_result}")
    assert index_result["status"] == "success", "Indexing failed!"
    assert index_result["chunks_indexed"] > 0, "No chunks were indexed!"
    assert index_result["pages_indexed"] == 3, f"Expected 3 pages, got {index_result['pages_indexed']}"
    print("   ✅ PDF Ingestion & Chunking Passed!")

    # Step 2: Test Retrieval Grounding & Page Preservation
    print("\n2. Testing Vector Retrieval & Page Number Preservation...")
    query = "What is the function of mitochondria and cristae?"
    retrieved = retrieve_relevant_chunks(query, top_k=4)
    assert len(retrieved) > 0, "No chunks retrieved!"
    top_chunk = retrieved[0]
    print(f"   Top retrieved chunk: Doc={top_chunk['doc_name']}, Page={top_chunk['page_num']}, Sim={top_chunk['similarity']}")
    assert top_chunk["doc_name"] == "sample_biology_notes.pdf"
    assert top_chunk["page_num"] == 1, f"Expected Page 1 for mitochondria, got Page {top_chunk['page_num']}"
    assert "mitochondria" in top_chunk["text"].lower() or "atp" in top_chunk["text"].lower()
    print("   ✅ Page-Level Retrieval Accuracy Passed!")

    # Step 3: Test DNA Replication Query (Page 3)
    print("\n3. Testing Query for Later Pages (Page 3 - DNA Replication)...")
    query_dna = "What is the role of DNA Helicase and DNA Ligase in replication?"
    retrieved_dna = retrieve_relevant_chunks(query_dna, top_k=2)
    assert len(retrieved_dna) > 0
    top_dna = retrieved_dna[0]
    print(f"   Top retrieved chunk: Doc={top_dna['doc_name']}, Page={top_dna['page_num']}, Sim={top_dna['similarity']}")
    assert top_dna["page_num"] == 3, f"Expected Page 3 for DNA replication, got Page {top_dna['page_num']}"
    assert "helicase" in top_dna["text"].lower() or "ligase" in top_dna["text"].lower()
    print("   ✅ Multi-Page Chunking & Retrieval Passed!")

    # Step 4: Test Answer Generation & Source Verification
    print("\n4. Testing Answer Generation & Source Structuring...")
    ans_res = answer_question("Explain the role of RuBisCO in the Calvin cycle")
    print(f"   Answer preview: {ans_res['answer'][:150]}...")
    assert len(ans_res["sources"]) > 0, "Expected sources to be returned!"
    assert any(s["page_num"] == 2 for s in ans_res["sources"]), "Expected Page 2 citation for Calvin cycle!"
    print("   ✅ Grounded Sources Structure Passed!")

    # Step 5: Test Abstention for Out-of-Scope Queries
    print("\n5. Testing Abstention on Out-of-Scope Questions...")
    out_of_scope_res = answer_question("What is the capital of Australia and who is the prime minister?")
    print(f"   Response: {out_of_scope_res['answer']}")
    assert "not contain" in out_of_scope_res["answer"].lower() or "unavailable" in out_of_scope_res["answer"].lower(), "System should abstain on out-of-scope question!"
    print("   ✅ Abstention & Non-Hallucination Passed!")

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! StudyMate AI is 100% verified and functional.")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
