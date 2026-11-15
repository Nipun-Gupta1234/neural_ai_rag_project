import os
import json
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from ingest import get_collection, get_embedding_model

load_dotenv()

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

# Google retires model names over time (the "-latest" aliases were removed too),
# so we keep a short list of known-good models and also fall back to whatever
# the API key reports as available.
DEFAULT_MODEL = "gemini-3.5-flash"
FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-2.5-flash"]


def _is_model_not_found(exc: Exception) -> bool:
    """True when the API rejected the model name (retired / unknown), not the key or quota."""
    return getattr(exc, "code", None) == 404 or "404" in str(exc)


def _is_model_overloaded(exc: Exception) -> bool:
    """True for temporary 'high demand' errors where another model may still work."""
    return getattr(exc, "code", None) == 503 or "503" in str(exc) or "UNAVAILABLE" in str(exc)


def list_available_models(api_key: str) -> List[str]:
    """
    Returns the Gemini text models ("flash" / "pro") this API key can use,
    newest version first. Returns [] if the list can't be fetched.
    """
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        names = []
        for m in client.models.list():
            actions = getattr(m, "supported_actions", None) or []
            name = (m.name or "").removeprefix("models/")
            is_text_model = ("flash" in name or "pro" in name) and not any(
                skip in name for skip in ("image", "embedding", "tts", "audio", "live", "preview")
            )
            if "generateContent" in actions and name.startswith("gemini-") and is_text_model:
                names.append(name)

        def version_key(n: str):
            nums = re.findall(r"\d+", n)
            return [int(x) for x in nums]

        return sorted(set(names), key=version_key, reverse=True)
    except Exception:
        return []


def call_gemini_llm(
    prompt: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    system_instruction: str = SYSTEM_PROMPT,
    json_output: bool = False
) -> str:
    """
    Calls Gemini through the google-genai SDK.

    If the chosen model has been retired (HTTP 404) or is temporarily
    overloaded (HTTP 503) it retries with fallback models, including the
    ones the API key reports as available. Any other error (bad key, quota,
    network) is raised immediately so the user sees the real cause.
    """
    effective_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not effective_key:
        raise ValueError("Missing Gemini API Key. Please provide an API key.")

    from google import genai
    from google.genai import types

    primary_model = model_name or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
    candidate_models = [primary_model] + [m for m in FALLBACK_MODELS if m != primary_model]

    client = genai.Client(api_key=effective_key)
    listed_models_added = False
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.2,
        response_mime_type="application/json" if json_output else None
    )

    not_found_errors = []
    overloaded_errors = []
    i = 0
    while i < len(candidate_models):
        cand = candidate_models[i]
        i += 1
        try:
            response = client.models.generate_content(model=cand, contents=prompt, config=config)
        except Exception as e:
            if _is_model_not_found(e) or _is_model_overloaded(e):
                (not_found_errors if _is_model_not_found(e) else overloaded_errors).append(cand)
                # Once the built-in list is exhausted, try the "flash" models this key can see.
                if i == len(candidate_models) and not listed_models_added:
                    listed_models_added = True
                    extra = [m for m in list_available_models(effective_key)
                             if "flash" in m and "image" not in m and m not in candidate_models]
                    candidate_models.extend(extra[:5])
                continue
            raise
        text = (response.text or "").strip() if response else ""
        if not text:
            raise RuntimeError(f"Gemini model '{cand}' returned an empty response (it may have been blocked by safety filters).")
        return text

    parts = []
    if not_found_errors:
        parts.append(f"not available to your API key: {', '.join(not_found_errors)}")
    if overloaded_errors:
        parts.append(f"temporarily overloaded (try again in a minute): {', '.join(overloaded_errors)}")
    raise RuntimeError(
        "No Gemini model could answer. Models " + "; ".join(parts) + ". "
        "Pick a different model in the sidebar or set GEMINI_MODEL in .env."
    )

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

    raw_text = call_gemini_llm(
        prompt=prompt,
        api_key=effective_key,
        model_name=model_name,
        system_instruction=SYSTEM_PROMPT,
        json_output=True
    )
    return parse_mcqs(raw_text)


def parse_mcqs(raw_text: str) -> List[Dict[str, Any]]:
    """
    Parses the model's JSON reply into a list of valid MCQs.
    Questions missing a question, 4 options, or an A-D answer are dropped.
    Raises ValueError if the reply contains no usable questions.
    """
    # Strip code fences in case the model added them anyway
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip())
    # Fall back to the outermost [...] if there is extra text around the JSON
    if not text.startswith("["):
        match = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if match:
            text = match.group(0)

    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("questions", [data])

    valid = []
    for q in data if isinstance(data, list) else []:
        if not isinstance(q, dict):
            continue
        options = q.get("options")
        answer = str(q.get("answer", "")).strip()[:1].upper()
        if q.get("question") and isinstance(options, list) and len(options) == 4 and answer in "ABCD" and answer:
            q["answer"] = answer
            valid.append(q)

    if not valid:
        raise ValueError("Gemini's reply did not contain any valid quiz questions.")
    return valid
