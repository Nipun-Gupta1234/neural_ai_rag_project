# 🎓 StudyMate AI — RAG-Powered PDF Study Notes Assistant & Quiz Generator

**StudyMate AI** is a lightweight, fully functional Retrieval-Augmented Generation (RAG) web application designed for students and educators. It turns uploaded PDF lecture notes and textbooks into an interactive AI tutor and custom quiz generator with verifiable, page-level citations.

---

## 🌟 Key Features

1. **Multi-PDF Upload & Ingestion**:
   - Upload one or multiple course notes, lecture slides, or textbook PDFs.
   - Extracts clean text using **PyMuPDF**, strictly preserving source document names and 1-indexed page numbers.
2. **Local Vector Storage & Embeddings**:
   - Uses **Sentence Transformers (`all-MiniLM-L6-v2`)** for local, fast 384-dimensional dense semantic embeddings.
   - Embeddings and metadata are persisted locally in **ChromaDB** without requiring paid vector database subscriptions.
3. **Strictly Grounded Q&A Chat**:
   - Retrieves the top 4 most relevant chunks for each student question.
   - Prompts the **Gemini API** with strict instructions to answer *only* from the retrieved study excerpts.
   - Multi-turn conversational interface with chat history in Streamlit session state.
4. **Verifiable Source References**:
   - Displays real citations under each answer showing Document Name, Page Number, and excerpt snippet.
5. **Interactive 5-Question MCQ Quiz Generator**:
   - Generates 5 multiple-choice questions grounded in the uploaded notes.
   - Interactive quiz interface with radio button selections.
   - Calculates total score and provides detailed explanations citing the source document and page number.
6. **Hallucination Prevention & Abstention**:
   - If the uploaded study notes do not contain sufficient evidence to answer the question, StudyMate AI explicitly says so rather than fabricating information.
7. **Secure API Key Management**:
   - Loads the Gemini API key from `.env` or allows direct input in the sidebar.

---

## 🏗️ Project Structure

```
neural_ai_rag_project/
├── app.py                      # Main Streamlit web application
├── ingest.py                   # PDF text extraction, chunking, and ChromaDB indexing
├── rag.py                      # Vector retrieval, Gemini answer generation, and MCQ quiz logic
├── data/
│   └── sample_biology_notes.pdf # Included sample study notes for instant testing
├── chroma_db/                  # Local persistent ChromaDB vector storage (auto-created)
├── requirements.txt            # Python dependencies
├── .env.example                # Example environment variables
├── .env                        # Local API key file
└── README.md                   # Complete documentation and setup guide
```

---

## 🚀 Setup & Execution Guide (Windows 11 / PowerShell)

### Prerequisites
- Python 3.11+ or 3.12+ installed on Windows.
- A free Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/).

### Step 1: Clone or Navigate to Project Directory
```powershell
cd "d:\GitHub Repos\neural_ai_rag_project"
```

### Step 2: Create and Activate Python Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Configure Your Gemini API Key
Create a `.env` file (or copy from `.env.example`):
```powershell
Copy-Item .env.example .env
```
Edit `.env` and paste your Gemini API key:
```ini
GEMINI_API_KEY=your_actual_key_here
GEMINI_MODEL=gemini-1.5-flash
```
*(Alternatively, you can also paste your Gemini API key directly into the Streamlit sidebar field.)*

### Step 5: Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

---

## 🧪 Sample Questions for Instant Testing

A multi-page sample textbook PDF is pre-generated in `data/sample_biology_notes.pdf` covering Cell Biology, Photosynthesis, and DNA Replication.

Once you upload and index `data/sample_biology_notes.pdf`, try asking these questions:

### Test Case 1: Specific Organelle Function (Page 1)
- **Question:** *"What is the main function of the mitochondria and why is its inner membrane folded into cristae?"*
- **Expected Grounding:** Explains ATP generation via oxidative phosphorylation and that cristae increase surface area for ATP synthase complexes (Page 1).

### Test Case 2: Photosynthesis & Calvin Cycle (Page 2)
- **Question:** *"What happens during the light-dependent reactions and which enzyme fixes CO2 in the Calvin cycle?"*
- **Expected Grounding:** Identifies photolysis of water into 2H+, 1/2 O2, and electrons, and names RuBisCO enzyme (Page 2).

### Test Case 3: DNA Replication Enzymes (Page 3)
- **Question:** *"What are the roles of DNA Helicase and DNA Ligase during DNA replication?"*
- **Expected Grounding:** Cites DNA Helicase unwinding the double helix and DNA Ligase sealing Okazaki fragments on the lagging strand (Page 3).

### Test Case 4: Abstention / Out-of-Scope Question
- **Question:** *"What is the capital of Australia and who is the prime minister?"*
- **Expected Grounding:** The system explicitly states that the uploaded notes do not contain this information rather than hallucinating.

### Test Case 5: 5-Question MCQ Quiz
- Navigate to the **"📝 AI Quiz Generator"** tab and click **"Generate 5 MCQs"**.
- Answer the interactive quiz, click **"Submit Quiz Answers"**, and verify your score, correct answers, and explanations with page citations.

---

## 🛡️ Architecture & Constraints Followed

- **Zero Unnecessary Bloat**: Built purely with Python, Streamlit, ChromaDB, Sentence Transformers, and PyMuPDF (no Docker, React, or complex backends).
- **Persistent & Reusable**: Indexed vectors are stored locally in `./chroma_db` so you index your PDFs once and query them across sessions.
- **Local Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`) runs locally on your machine without external API charges for embedding.
- **Verifiable Citations**: Every single answer displays the document name and page number from which the excerpt was retrieved.
