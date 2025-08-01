# ai_utils.py

import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
import google.generativeai as genai

# === CONFIG ===
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
genai_model = genai.GenerativeModel("gemini-2.5-flash")

# === STATE (in-memory storage for demo) ===
video_cache = {}  # Stores transcript, chunks, index, embeddings for each video


# === CORE FUNCTIONS ===

def process_text(transcript, video_id=None):
    """
    Process the transcript text into summary, flashcards, and vector index.
    """
    doc = Document(page_content=transcript, metadata={"source": video_id or "unknown"})
    docs = text_splitter.split_documents([doc])
    chunks = [d.page_content for d in docs]

    # Embedding and FAISS index
    vectors = np.array(embed_model.encode(chunks))
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    # Store for later queries
    if video_id:
        video_cache[video_id] = {
            "chunks": chunks,
            "index": index,
            "embeddings": vectors
        }

    # Generate Summary
    summary_prompt = f"Summarize the following transcript:\n\n{transcript}"
    summary = genai_model.generate_content(summary_prompt).text.strip()

    # Generate Flashcards
    card_prompt = f\"\"\"\nYou are an AI assistant. Generate 10 flashcards from this transcript:\n\n{transcript}\n\nFormat:\nQ: ...\nA: ...\n\"\"\"\n    flashcards = genai_model.generate_content(card_prompt).text.strip()\n\n    return summary, flashcards, index\n\ndef query_answer(question, video_id):\n    if video_id not in video_cache:\n        return \"❌ Video context not found. Please analyze the video first.\"\n\n    chunks = video_cache[video_id][\"chunks\"]\n    index = video_cache[video_id][\"index\"]\n    embeddings = video_cache[video_id][\"embeddings\"]\n\n    # Embed the query\n    q_vec = np.array(embed_model.encode([question]))\n    distances, indices = index.search(q_vec, k=4)\n    context_chunks = [chunks[i] for i in indices[0]]\n    context = \"\\n\".join(context_chunks)\n\n    prompt = f\"\"\"\nUse the following video transcript context to answer:\n\n=== CONTEXT START ===\n{context}\n=== CONTEXT END ===\n\nQ: {question}\nA:\n\"\"\"\n    answer = genai_model.generate_content(prompt).text.strip()\n    return answer\n```

---

### 🔍 Breakdown

| Function | Responsibility |
|---------|----------------|
| `process_text` | Splits transcript → embeds chunks → builds FAISS → stores video state → returns summary & flashcards |
| `query_answer` | Uses stored video chunks & FAISS index to fetch relevant context → sends to Gemini for answer |

---

### 🧠 Design Notes
- `video_cache` is RAM-only for MVP. Use Redis or DB for production.
- Stateless design can be added with unique hash IDs for caching videos.
- You can optionally return the audio file from `tts_utils.generate_tts()` here too.

Want me to modularize it further, or add a Redis/SQLite backend for `video_cache` persistence?
