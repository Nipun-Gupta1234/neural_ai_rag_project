import os
import json
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from ingest import get_collection, get_embedding_model

load_dotenv(override=True)

SYSTEM_PROMPT = """You are StudyMate AI, an expert, encouraging, and accurate AI study tutor.
Your job is to help students learn and understand their course materials based strictly on the provided study notes.

Rules:
1. Grounding: Answer strictly and faithfully using only the provided context from the student's study notes.
2. Citations: Attribute key concepts to the respective document and page number.
3. Honesty & Abstention: If the answer cannot be found in or deduced directly from the provided excerpts, clearly state:
   "The uploaded study notes do not contain sufficient information to answer this question. Please upload the relevant chapter or consult your textbook."
   Do NOT fabricate information or use external unverified knowledge.
4. Tone: Clear, well-structured, student-friendly, and educational. Use bullet points or numbered lists where appropriate for readability.
"""

def retrieve_relevant_chunks(query: str, top_k: int = 4, persist_dir: str = "chroma_db") -> List[Dict[str, Any]]:
    """
    Retrieves the top_k most relevant chunks from ChromaDB for a given query
    using the local all-MiniLM-L6-v2 embedding model.
    """
    collection = get_collection(persist_dir)
    if collection.count() == 0:
        return []

    model = get_embedding_model()
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    retrieved_chunks = []
    if results and results["ids"] and len(results["ids"][0]) > 0:
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, doc_text, meta, dist in zip(ids, documents, metadatas, distances):
            similarity = max(0.0, 1.0 - dist)
            retrieved_chunks.append({
                "chunk_id": chunk_id,
                "text": doc_text,
                "doc_name": meta.get("doc_name", "Unknown Document"),
                "page_num": meta.get("page_num", 1),
                "similarity": round(float(similarity), 4)
            })

    return retrieved_chunks

def format_context_for_prompt(chunks: List[Dict[str, Any]]) -> str:
    """Formats retrieved chunks into clean numbered context blocks."""
    formatted = []
    for idx, c in enumerate(chunks, 1):
        formatted.append(
            f"--- Excerpt {idx} [Document: {c['doc_name']} | Page: {c['page_num']}] ---\n"
            f"{c['text']}\n"
        )
    return "\n".join(formatted)

def get_available_models(api_key: Optional[str] = None) -> List[str]:
    """
    Fetches the actual list of available models for the given API key
    using ModelService.ListModels.
    """
    effective_key = (api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
    if not effective_key or effective_key == "your_gemini_api_key_here":
        return []

    # Try modern google.genai SDK across v1beta and v1
    try:
        from google import genai
        from google.genai import types
        for api_ver in ["v1beta", "v1"]:
            try:
                client = genai.Client(
                    api_key=effective_key,
                    http_options=types.HttpOptions(api_version=api_ver)
                )
                models = []
                for m in client.models.list():
                    actions = getattr(m, "supported_actions", []) or []
                    if "generateContent" in actions or not actions:
                        name = m.name.replace("models/", "") if m.name else ""
                        if name and "gemini" in name.lower() and "embedding" not in name.lower():
                            models.append(name)
                if models:
                    return models
            except Exception:
                continue
    except Exception:
        pass

    # Fallback to legacy google.generativeai
    try:
        import google.generativeai as legacy_genai
        legacy_genai.configure(api_key=effective_key)
        models = []
        for m in legacy_genai.list_models():
            methods = getattr(m, "supported_generation_methods", []) or []
            if "generateContent" in methods:
                name = m.name.replace("models/", "") if m.name else ""
                if name and "gemini" in name.lower() and "embedding" not in name.lower():
                    models.append(name)
        if models:
            return models
    except Exception:
        pass

    return []

def call_gemini_llm(
    prompt: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    system_instruction: str = SYSTEM_PROMPT
) -> str:
    """
    Invokes the Gemini API dynamically discovering active models for the user's
    API key and testing both v1beta and v1 endpoints to prevent 404 errors.
    """
    effective_key = (api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
    if not effective_key or effective_key == "your_gemini_api_key_here":
        raise ValueError("Missing or placeholder Gemini API Key. Please enter your API key in the sidebar or .env file.")

    # 1. Query available models dynamically for this key
    available_models = get_available_models(effective_key)

    primary_model = (model_name or os.environ.get("GEMINI_MODEL", "")).strip()
    
    # Build candidate queue: user preference -> discovered models -> common defaults
    candidate_models = []
    if primary_model:
        candidate_models.append(primary_model)
    for m in available_models:
        if m not in candidate_models:
            candidate_models.append(m)
    for fallback in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]:
        if fallback not in candidate_models:
            candidate_models.append(fallback)

    last_error = None

    # Try modern google.genai across both v1beta and v1 endpoints
    try:
        from google import genai
        from google.genai import types

        for api_ver in ["v1beta", "v1"]:
            client = genai.Client(
                api_key=effective_key,
                http_options=types.HttpOptions(api_version=api_ver)
            )
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2
            )

            for cand in candidate_models:
                try:
                    response = client.models.generate_content(
                        model=cand,
                        contents=prompt,
                        config=config
                    )
                    if response and response.text:
                        return response.text.strip()
                except Exception as e:
                    last_error = e
                    err_str = str(e).lower()
                    if "404" in err_str or "not found" in err_str or "not supported" in err_str:
                        continue
                    # Non-404 errors (like auth, quota, billing) are fatal, re-raise immediately
                    raise e
    except Exception as e:
        last_error = e

    # Fallback: legacy google.generativeai
    try:
        import google.generativeai as legacy_genai
        legacy_genai.configure(api_key=effective_key)
        for cand in candidate_models:
            try:
                model = legacy_genai.GenerativeModel(
                    model_name=cand,
                    system_instruction=system_instruction
                )
                response = model.generate_content(prompt)
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                last_error = e
                continue
    except Exception as e:
        last_error = e

    if last_error:
        # Provide helpful diagnosis if all models returned 404
        if "404" in str(last_error):
            discovered_str = ", ".join(available_models) if available_models else "None"
            raise RuntimeError(
                f"None of the requested Gemini models were accessible on this API key (Available for your key: [{discovered_str}]). "
                f"Please ensure the 'Generative Language API' is enabled in your Google Cloud / AI Studio project, "
                f"or generate a new key at https://aistudio.google.com/."
            )
        raise last_error

    raise RuntimeError("Failed to generate content with Gemini API.")

def get_gemini_model(api_key: Optional[str] = None, model_name: Optional[str] = None):
    """Backwards-compatibility stub."""
    effective_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not effective_key:
        return None
    return True

def answer_question(
    query: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    top_k: int = 4
) -> Dict[str, Any]:
    """
    Executes end-to-end RAG:
    1. Retrieves top_k chunks from ChromaDB
    2. Builds context
    3. Calls Gemini API for grounded generation
    4. Returns answer and structured sources
    """
    chunks = retrieve_relevant_chunks(query, top_k=top_k)

    if not chunks:
        return {
            "answer": "No study notes have been uploaded or indexed yet. Please upload your PDF notes in the sidebar to get started!",
            "sources": []
        }

    # Format sources
    sources = [{
        "doc_name": c["doc_name"],
        "page_num": c["page_num"],
        "snippet": c["text"][:180] + ("..." if len(c["text"]) > 180 else ""),
        "similarity": c["similarity"]
    } for c in chunks]

    # Check relevance threshold (abstention check)
    max_sim = max(c["similarity"] for c in chunks)
    if max_sim < 0.20:
        return {
            "answer": (
                "The uploaded study notes do not contain sufficient information to answer this question. "
                "Please check if you have uploaded the relevant chapter or textbook pages."
            ),
            "sources": []
        }

    effective_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not effective_key:
        # Fallback informative preview if API key is not yet set
        sources_summary = "\n".join([f"- **{s['doc_name']}** (Page {s['page_num']})" for s in sources])
        return {
            "answer": (
                "⚠️ **Gemini API Key Required**\n\n"
                "To generate natural language answers with AI, please enter your Gemini API key in the sidebar or add it to `.env`.\n\n"
                f"**Retrieved {len(chunks)} relevant excerpt(s) from your notes:**\n"
                f"{sources_summary}\n\n"
                f"**Top Excerpt Preview:**\n> {chunks[0]['text'][:300]}..."
            ),
            "sources": sources
        }

    context_str = format_context_for_prompt(chunks)
    prompt = (
        f"Study Notes Excerpts:\n{context_str}\n\n"
        f"Student Question: {query}\n\n"
        f"Please provide an accurate, helpful answer grounded strictly in the excerpts above. "
        f"If the answer is not contained in the excerpts, clearly state that the information is unavailable in the notes."
    )

    try:
        answer_text = call_gemini_llm(
            prompt=prompt,
            api_key=effective_key,
            model_name=model_name,
            system_instruction=SYSTEM_PROMPT
        )
        return {
            "answer": answer_text,
            "sources": sources
        }
    except Exception as e:
        return {
            "answer": f"Error generating answer from Gemini API: {str(e)}. Please check your API key and connection.",
            "sources": sources
        }

def generate_quiz(
    topic_or_query: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    num_questions: int = 5
) -> List[Dict[str, Any]]:
    """
    Generates 5 multiple choice questions grounded in the uploaded notes.
    """
    search_q = topic_or_query or "key concepts definitions principles laws summary important facts"
    chunks = retrieve_relevant_chunks(search_q, top_k=6)

    if not chunks:
        return []

    effective_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not effective_key:
        return []

    context_str = format_context_for_prompt(chunks)
    prompt = (
        f"Study Notes Excerpts:\n{context_str}\n\n"
        f"Generate exactly {num_questions} Multiple Choice Questions (MCQs) for students based strictly on the study notes excerpts above.\n"
        f"Each question must test an actual fact or concept from the notes.\n"
        f"Respond ONLY with a valid JSON array of objects. Do not include markdown code block formatting or extra conversational text.\n"
        f"Schema for each question:\n"
        f"[\n"
        f"  {{\n"
        f"    \"question\": \"Question text here?\",\n"
        f"    \"options\": [\"A) Option text\", \"B) Option text\", \"C) Option text\", \"D) Option text\"],\n"
        f"    \"answer\": \"A\",\n"
        f"    \"explanation\": \"Brief explanation citing the document context.\",\n"
        f"    \"source_doc\": \"Document name\",\n"
        f"    \"page_num\": 1\n"
        f"  }}\n"
        f"]"
    )

    try:
        raw_text = call_gemini_llm(
            prompt=prompt,
            api_key=effective_key,
            model_name=model_name,
            system_instruction=SYSTEM_PROMPT
        )

        # Clean code fences if present
        raw_text = re.sub(r"^```(?:json)?", "", raw_text, flags=re.MULTILINE)
        raw_text = re.sub(r"```$", "", raw_text, flags=re.MULTILINE).strip()

        mcqs = json.loads(raw_text)
        return mcqs
    except Exception as e:
        print(f"Error generating quiz: {e}")
        return []
