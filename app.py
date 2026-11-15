import os
import importlib
import streamlit as st
from dotenv import load_dotenv

# Ensure environment variables are reloaded from .env on each run
load_dotenv(override=True)

import ingest
import rag
importlib.reload(ingest)
importlib.reload(rag)

from ingest import index_pdf_files, clear_knowledge_base, get_collection
from rag import answer_question, generate_quiz, retrieve_relevant_chunks, get_available_models

st.set_page_config(
    page_title="StudyMate AI — Student Study Companion",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .source-box {
        background-color: #f8fafc;
        border-left: 4px solid #0d9488;
        padding: 0.6rem 0.8rem;
        border-radius: 0.4rem;
        margin-top: 0.5rem;
        font-size: 0.82rem;
    }
    .score-banner {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        padding: 1rem;
        border-radius: 0.6rem;
        color: #166534;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = []

if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("### 🎓 StudyMate AI")
    st.caption("RAG-Powered Student Notes Tutor & Quiz Generator")
    st.divider()

    # API Key Handling
    env_api_key = os.environ.get("GEMINI_API_KEY", "")
    api_key_input = st.text_input(
        "Gemini API Key",
        value=env_api_key,
        type="password",
        help="Loads automatically from .env or enter here"
    )
    api_key = api_key_input.strip() if api_key_input else None

    # Dynamic Model Selection
    discovered_models = []
    if api_key and len(api_key) > 10:
        try:
            discovered_models = get_available_models(api_key)
        except Exception:
            discovered_models = []

    model_options = discovered_models if discovered_models else [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro"
    ]
    env_model = os.environ.get("GEMINI_MODEL", model_options[0])
    default_idx = model_options.index(env_model) if env_model in model_options else 0
    selected_model = st.selectbox(
        "Gemini Model",
        options=model_options,
        index=default_idx,
        help="Select the Gemini model variant. Dynamically populated based on your API key."
    )

    st.divider()

    # PDF Upload
    st.markdown("#### 📂 Upload Notes")
    uploaded_files = st.file_uploader(
        "Upload study notes (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or multiple lecture notes, book chapters, or revision sheets."
    )

    if st.button("⚡ Index Documents", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Please select at least one PDF file to upload.")
        else:
            with st.spinner("Extracting text and generating embeddings..."):
                files_payload = [(f.name, f.read()) for f in uploaded_files]
                result = index_pdf_files(files_payload)
                if result["status"] == "success":
                    st.success(f"Indexed {result['chunks_indexed']} chunks from {result['docs_indexed']} PDF(s)!")
                else:
                    st.error(result["message"])

    # Current Index Status
    collection = get_collection()
    chunk_count = collection.count()
    st.markdown(f"**📚 Knowledge Base Status:** `{chunk_count}` chunk(s) indexed")

    if chunk_count > 0:
        if st.button("🗑️ Reset Notes Index", use_container_width=True):
            clear_knowledge_base()
            st.session_state.messages = []
            st.session_state.quiz_data = []
            st.session_state.quiz_submitted = False
            st.success("Knowledge base cleared!")
            st.rerun()

    st.divider()
    st.caption("Powered by ChromaDB • all-MiniLM-L6-v2 • Gemini API")


# ----------------- MAIN TABS -----------------
st.markdown('<div class="main-header">StudyMate AI 🎓</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions about your uploaded study notes with real page citations, or test your knowledge with auto-generated MCQ quizzes.</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["💬 Ask Your Notes", "📝 AI Quiz Generator", "🔍 Document Explorer"])

# ----------------- TAB 1: CHATBOT -----------------
with tab1:
    col_q, col_clear = st.columns([5, 1])
    with col_clear:
        if st.button("🧹 Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # Sample Quick Prompts
    st.markdown("**Suggested Questions:**")
    quick_cols = st.columns(3)
    sample_queries = [
        "What are the core concepts explained in the document?",
        "Provide a summary of the main definitions.",
        "What formulas or key facts are highlighted?"
    ]

    selected_quick_q = None
    for idx, sq in enumerate(sample_queries):
        if quick_cols[idx].button(f"💡 {sq}", key=f"sq_{idx}"):
            selected_quick_q = sq

    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 Sources & Citations ({len(msg['sources'])})"):
                    for s in msg["sources"]:
                        st.markdown(
                            f"**📄 {s['doc_name']} — Page {s['page_num']}** "
                            f"`Similarity: {s['similarity']}`\n\n"
                            f"> {s['snippet']}\n"
                        )

    # Chat Input Handling
    chat_prompt = st.chat_input("Ask a question about your study materials...")
    query_to_run = selected_quick_q or chat_prompt

    if query_to_run:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": query_to_run})
        with st.chat_message("user"):
            st.markdown(query_to_run)

        # Generate Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching notes and generating answer..."):
                response = answer_question(query_to_run, api_key=api_key, model_name=selected_model, top_k=4)
                st.markdown(response["answer"])

                if response.get("sources"):
                    with st.expander(f"📚 Sources & Citations ({len(response['sources'])})"):
                        for s in response["sources"]:
                            st.markdown(
                                f"**📄 {s['doc_name']} — Page {s['page_num']}** "
                                f"`Similarity: {s['similarity']}`\n\n"
                                f"> {s['snippet']}\n"
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response.get("sources", [])
                })


# ----------------- TAB 2: QUIZ GENERATOR -----------------
with tab2:
    st.markdown("### 📝 Test Your Mastery: 5-Question MCQ Quiz")
    st.caption("StudyMate AI extracts key concepts from your uploaded notes and crafts a custom multiple-choice test.")

    quiz_topic = st.text_input("Quiz Topic or Focus Area (Optional)", placeholder="e.g. Chapter 1, Newton's Laws, Cell Biology, General Review")

    col_btn1, _ = st.columns([2, 4])
    with col_btn1:
        if st.button("✨ Generate 5 MCQs", type="primary", use_container_width=True):
            if chunk_count == 0:
                st.warning("Please upload and index your study PDF(s) in the sidebar first!")
            elif not api_key:
                st.warning("Gemini API Key required to generate custom quizzes.")
            else:
                with st.spinner("Analyzing notes and generating questions..."):
                    mcqs = generate_quiz(topic_or_query=quiz_topic, api_key=api_key, model_name=selected_model, num_questions=5)
                    if mcqs:
                        st.session_state.quiz_data = mcqs
                        st.session_state.quiz_submitted = False
                        st.session_state.user_answers = {}
                        st.success("Quiz generated successfully!")
                    else:
                        st.error("Could not generate quiz. Please check API key and try again.")

    # Render Quiz
    if st.session_state.quiz_data:
        st.divider()
        mcqs = st.session_state.quiz_data

        with st.form("mcq_quiz_form"):
            for idx, q in enumerate(mcqs, 1):
                st.markdown(f"**Q{idx}. {q['question']}**")
                options = q.get("options", [])
                
                # Option format: ["A) ...", "B) ...", "C) ...", "D) ..."]
                choice = st.radio(
                    f"Select answer for Q{idx}:",
                    options=options,
                    key=f"q_choice_{idx}",
                    label_visibility="collapsed"
                )
                st.session_state.user_answers[idx] = choice
                st.write("")

            submitted = st.form_submit_button("Submit Quiz Answers", type="primary")
            if submitted:
                st.session_state.quiz_submitted = True

        # Results & Explanations
        if st.session_state.quiz_submitted:
            st.divider()
            correct_count = 0

            for idx, q in enumerate(mcqs, 1):
                user_selected = st.session_state.user_answers.get(idx, "")
                # Extract letter prefix 'A', 'B', 'C', 'D'
                user_letter = user_selected[:1].upper() if user_selected else ""
                correct_letter = str(q.get("answer", "")).strip()[:1].upper()

                is_correct = (user_letter == correct_letter)
                if is_correct:
                    correct_count += 1

            # Score Banner
            score_pct = int((correct_count / len(mcqs)) * 100)
            st.markdown(
                f'<div class="score-banner">🎉 Your Score: {correct_count} / {len(mcqs)} ({score_pct}%)</div>',
                unsafe_allow_html=True
            )

            # Detailed review
            for idx, q in enumerate(mcqs, 1):
                user_selected = st.session_state.user_answers.get(idx, "")
                user_letter = user_selected[:1].upper() if user_selected else ""
                correct_letter = str(q.get("answer", "")).strip()[:1].upper()
                is_correct = (user_letter == correct_letter)

                with st.expander(
                    f"Q{idx}: {'✅ Correct' if is_correct else '❌ Incorrect'} — {q['question']}",
                    expanded=not is_correct
                ):
                    st.markdown(f"**Your Answer:** `{user_selected or 'None'}`")
                    st.markdown(f"**Correct Answer:** Option **{correct_letter}**")
                    st.markdown(f"**💡 Explanation:** {q.get('explanation', 'Derived from your uploaded study notes.')}")
                    if q.get("source_doc"):
                        st.caption(f"Source: {q['source_doc']} (Page {q.get('page_num', 'N/A')})")


# ----------------- TAB 3: EXPLORER -----------------
with tab3:
    st.markdown("### 🔍 Knowledge Base Chunks Explorer")
    st.caption("Inspect the active ChromaDB vector collection and chunk previews.")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Indexed Chunks", chunk_count)
    with col2:
        st.metric("Embedding Model", "all-MiniLM-L6-v2 (384 dimensions)")

    if chunk_count > 0:
        search_probe = st.text_input("Search chunks in database:", placeholder="Enter keyword to inspect retrieved chunks...")
        if search_probe:
            matches = retrieve_relevant_chunks(search_probe, top_k=4)
            st.markdown(f"**Top {len(matches)} matching chunks:**")
            for m in matches:
                with st.container():
                    st.markdown(f"**📄 {m['doc_name']} (Page {m['page_num']})** • `Cosine Sim: {m['similarity']}`")
                    st.info(m["text"])
    else:
        st.info("No study notes in database yet. Upload PDFs using the sidebar.")
