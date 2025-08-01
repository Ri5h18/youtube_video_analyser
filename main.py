from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import google.generativeai as genai
import faiss
import numpy as np
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from yt_dlp import YoutubeDL
import requests

# ========== CONFIG ==========
RAPIDAPI_KEY = "9bf414ebeemsh3266a2940239b68p1c2ff1jsn21df8793885c"
GEMINI_KEY = "AIzaSyD_6yIASHojj1Of_pKMtqHAhkQdwzhK3p0"

genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel("gemma-3-27b-it")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

# ========== FASTAPI APP ==========
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ========== FUNCTIONS ==========
def get_transcript(video_id):
    url = "https://youtube-v2.p.rapidapi.com/video/subtitles"
    querystring = {"video_id": video_id}
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "youtube-v2.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    if response.status_code != 200:
        return None
    data = response.json()
    subtitles = data.get("subtitles", [])
    return " ".join(sub["text"] for sub in subtitles)

def embed_chunks(chunks):
    vectors = np.array(embed_model.encode(chunks))
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index, vectors

def retrieve_relevant(query, chunks, index, embeddings, k=4):
    q_embed = np.array(embed_model.encode([query]))
    distances, indices = index.search(q_embed, k)
    return [chunks[i] for i in indices[0]]

def generate_answer(query, context_chunks):
    context = "\n".join(chunk.page_content if hasattr(chunk, "page_content") else str(chunk) for chunk in context_chunks)
    prompt = f"""You are a helpful and knowledgeable assistant. Answer the following question using both the provided context and your own knowledge. Do not separate them — synthesize everything into a coherent, natural response.

=== CONTEXT START ===
{context}
=== CONTEXT END ===

Question: {query}
Answer:"""
    result = model.generate_content(prompt)
    return result.text.strip()

def generate_flashcards(transcript_text, num_cards=10):
    prompt = f"""
You are an AI assistant. Given the following transcript, generate {num_cards} flashcards for learning.
Each flashcard should be in this format:

Q: <question>
A: <answer>

Avoid duplication, keep answers short and to the point.

Transcript:
\"\"\"{transcript_text}\"\"\"
"""
    result = model.generate_content(prompt)
    return result.text.strip()

# ========== ROUTES ==========
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
cache = {}

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, video_url: str = Form(...)):
    try:
        video_id = video_url.split("v=")[-1].split("&")[0]
        ydl_opts = {'quiet': True, 'skip_download': True, 'format': 'bestaudio/best'}
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            duration = info.get('duration', 0)
            mins, secs = divmod(duration, 60)
            upload_fmt = datetime.strptime(info.get('upload_date', '19700101'), "%Y%m%d").strftime("%B %d, %Y")

        text = get_transcript(video_id)
        if not text:
            return templates.TemplateResponse("error.ejs", {"request": request, "error_message": "Transcript not available."})

        doc = Document(page_content=text, metadata={"source": video_id})
        docs = text_splitter.split_documents([doc]) if duration > 8 * 60 else [doc]
        all_chunks = [doc.page_content for doc in docs]
        index, embeddings = embed_chunks(all_chunks)

        summary_prompt = f"Summarize the following transcript:\n\n{text}"
        summary = model.generate_content(summary_prompt).text.strip()
        flashcards = generate_flashcards(text)

        cache[video_id] = templates.TemplateResponse("result.html", {
            "request": request,
            "title": info.get("title"),
            "duration": f"{mins} min {secs} sec",
            "upload_date": upload_fmt,
            "summary": summary,
            "flashcards": flashcards,
            "video_id": video_id,
            "all_chunks": all_chunks,
            "index_obj": index,
            
        })
        return cache[video_id]
    except Exception as e:
        return templates.TemplateResponse("error.html", {"request": request, "error_message": str(e)})

@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, video_id: str = Form(...), query: str = Form(...)):
    transcript = get_transcript(video_id)
    doc = Document(page_content=transcript, metadata={"source": video_id})
    docs = text_splitter.split_documents([doc]) if len(transcript) > 3000 else [doc]
    all_chunks = [doc.page_content for doc in docs]
    index, embeddings = embed_chunks(all_chunks)
    relevant = retrieve_relevant(query, all_chunks, index, embeddings)
    answer = generate_answer(query, relevant)
    return templates.TemplateResponse("result.html", {
        "request": request,
        "title": "N/A",
        "duration": "N/A",
        "upload_date": "N/A",
        "summary": "",
        "flashcards": "",
        "answer": answer,
        "video_id": video_id
    })
