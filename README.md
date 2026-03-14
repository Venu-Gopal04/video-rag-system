# 🎥 Video RAG Query System

Query any video using natural language — like asking CCTV footage 
"Show me all forklift near-misses yesterday"

## 🚀 What it does
- Upload any video (MP4, AVI, MOV, MKV)
- AI automatically extracts frames every 2 seconds
- Vision LLM describes each frame in detail
- ChromaDB indexes all descriptions as vectors
- Ask any natural language question → get answers with exact timestamps

## 🛠️ Tech Stack
- **Backend:** FastAPI, Python
- **Frame Extraction:** OpenCV
- **AI Vision:** Groq LLaVA (frame descriptions)
- **Vector Store:** ChromaDB
- **LLM:** Groq LLaMA 3.3 70B (answer generation)
- **Frontend:** HTML, CSS, JavaScript

## 💡 Example Queries
- "Show me all forklift activity"
- "Who is not wearing a helmet?"
- "Any unsafe behaviors detected?"
- "Show workers near machinery"

## ⚙️ Setup
```bash
pip install -r requirements.txt
# Add GROQ_API_KEY to .env file
uvicorn main:app --reload
```
Open http://127.0.0.1:8000
