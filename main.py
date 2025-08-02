import logging
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import yt_dlp
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from yt_dlp import YoutubeDL
import numpy as np
import requests
import os
from dotenv import load_dotenv
from google.genai import types
from youtube_transcript_api import YouTubeTranscriptApi
# ========== LOGGING SETUP ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== CONFIG ==========
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_KEY")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

genai.configure(api_key=GEMINI_KEY)
from google import genai
client = genai.Client(api_key=GEMINI_KEY)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
templates = Jinja2Templates(directory="templates")
app = FastAPI()
# to prevent re embedding and etc
cache = {}

# 

def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        logger.info(transcript)
        return " ".join([item['text'] for item in transcript])
    except Exception as e:
        logger.error(f"Transcript error: {e}")
        return None
import time
def embed_chunks(chunks):
    # logger.info(chunks)
    logger.info(f"Embedding {len(chunks)} chunks")
    vectors = []
    for i, chunk in enumerate(chunks):
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunk,
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        logger.info(result)
        [embedding_obj] = result.embeddings
        vectors.append(embedding_obj.values)
        logger.debug(f"Embedded chunk {i}")
        time.sleep(1)
    return np.array(vectors)

def retrieve_relevant(query, chunks, embeddings, k=4):
    logger.info(f"Retrieving top {k} relevant chunks for query: {query}")
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        config=types.EmbedContentConfig(output_dimensionality=768)
    )
    logger.info(result)
    [query_embed] = result.embeddings
    query_vector = np.array(query_embed.values).reshape(1, -1)
    sims = cosine_similarity(query_vector, embeddings)[0]
    top_k = np.argsort(sims)[::-1][:k]
    logger.debug(f"Top indices: {top_k}")
    return [chunks[i] for i in top_k]

def generate_answer(query, context_chunks):
    logger.info("Generating answer for user query")
    context = "\n".join(chunk.page_content if hasattr(chunk, "page_content") else str(chunk) for chunk in context_chunks)
    prompt = f"""You are a helpful and knowledgeable assistant. Answer the following question using both the provided context and your own knowledge. Do not separate them — synthesize everything into a coherent, natural response.

=== CONTEXT START ===
{context}
=== CONTEXT END ===

Question: {query}
Answer:"""
    response = client.models.generate_content(
        model="models/gemini-2.0-flash",
        contents=prompt
    )
    logger.debug("Answer generated")
    return response.text.strip()

def generate_flashcards(transcript_text, num_cards=10):
    logger.info(f"Generating {num_cards} flashcards")
    prompt = f"""
You are an AI assistant. Given the following transcript, generate {num_cards} flashcards for learning.
Each flashcard should be in this format:

Q: <question>
A: <answer>

Avoid duplication, keep answers short and to the point.

Transcript:
\"\"\"{transcript_text}\"\"\"
"""
    response = client.models.generate_content(
        model="models/gemini-2.0-flash",
        contents=prompt
    )
    logger.debug("Flashcards generated")
    return response.text.strip()



def generate_additional_flashcards(transcript_text, existing_flashcards=None, num_cards=5):
    logger.info(f"Generating {num_cards} additional flashcards")

    prompt = f"""
You are an AI assistant. Based on the following transcript, generate {num_cards} new flashcards that are not duplicates or paraphrases of existing ones.

Each flashcard should follow the format:
Q: <question>
A: <answer>

Keep answers short and relevant.

Transcript:
\"\"\"{transcript_text}\"\"\"

Existing flashcards:
\"\"\"{existing_flashcards or 'None'}\"\"\"
"""

    response = client.models.generate_content(
        model="models/gemini-2.0-flash",
        contents=prompt
    )

    logger.debug("Additional flashcards generated")
    return response.text.strip()



# ========== ROUTES ==========

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    logger.info("GET / - Home page")
    return templates.TemplateResponse("index.html", {"request": request})



@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, video_url: str = Form(...)):
    try:
        logger.info(f"POST /analyze - URL: {video_url}")
        video_id = video_url.split("v=")[-1].split("&")[0]
        if video_id in cache:
            logger.info(f"Using cached results for video_id: {video_id}")
            return templates.TemplateResponse("result.html", {
        "request": request,
        "title": cache[video_id]["title"],
        "duration": cache[video_id]["duration"],
        "upload_date": cache[video_id]["upload_date"],
        "summary": cache[video_id]["summary"],
        "flashcards": cache[video_id]["flashcards"],
        "video_id": video_id
        })
        logger.info(f"Extracted video_id: {video_id}")

        ydl_opts = {'quiet': True, 'skip_download': True, 'format': 'bestaudio/best'}
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            duration = info.get('duration', 0)
            mins, secs = divmod(duration, 60)
            upload_fmt = datetime.strptime(info.get('upload_date', '19700101'), "%Y%m%d").strftime("%B %d, %Y")
            logger.info(f"Video info - Duration: {duration}, Uploaded: {upload_fmt}")

        text = get_transcript(video_id)
        if not text:
            logger.warning("Transcript not available")
            return templates.TemplateResponse("error.html", {"request": request, "error_message": "Transcript not available."})

        doc = Document(page_content=text, metadata={"source": video_id})
        docs = text_splitter.split_documents([doc]) if duration > 8 * 60 else [doc]
        all_chunks = [doc.page_content for doc in docs]
        embeddings = embed_chunks(all_chunks)

        summary_prompt = f"Summarize the following transcript:\n\n{text}"
        summary = client.models.generate_content(
            model="models/gemini-2.0-flash",
            contents=summary_prompt
        ).text.strip()

        flashcards = generate_flashcards(text)

        cache[video_id] = {
            "chunks": all_chunks,
            "embeddings": embeddings,
            "title": info.get("title"),
            "duration": f"{mins} min {secs} sec",
            "upload_date": upload_fmt,
            "summary": summary,
            "flashcards": flashcards
        }

        logger.info(f"Analysis complete for video_id: {video_id}")
        return templates.TemplateResponse("result.html", {
            "request": request,
            "title": cache[video_id]["title"],
            "duration": cache[video_id]["duration"],
            "upload_date": cache[video_id]["upload_date"],
            "summary": summary,
            "flashcards": flashcards,
            "video_id": video_id
        })

    except Exception as e:
        logger.exception("Error in /analyze route")
        return templates.TemplateResponse("error.html", {"request": request, "error_message": str(e)})

@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, video_id: str = Form(...), query: str = Form(...)):
    try:
        logger.info(f"POST /ask - video_id: {video_id}, query: {query}")
        if video_id not in cache:
            logger.warning("Session expired or video not analyzed")
            return templates.TemplateResponse("error.html", {"request": request, "error_message": "Session expired or not analyzed yet."})

        chunks = cache[video_id]["chunks"]
        embeddings = cache[video_id]["embeddings"]
        relevant = retrieve_relevant(query, chunks, embeddings)
        answer = generate_answer(query, relevant)

        logger.info("Answer generation complete")
        return templates.TemplateResponse("result.html", {
            "request": request,
            "title": cache[video_id]["title"],
            "duration": cache[video_id]["duration"],
            "upload_date": cache[video_id]["upload_date"],
            "summary": cache[video_id]["summary"],
            "flashcards": cache[video_id]["flashcards"],
            "answer": answer,
            "video_id": video_id
        })

    except Exception as e:
        logger.exception("Error in /ask route")
        return templates.TemplateResponse("error.html", {"request": request, "error_message": str(e)})


@app.post("/flashcards", response_class=HTMLResponse)
async def regenerate_flashcards(request: Request, video_id: str = Form(...)):
    try:
        logger.info(f"POST /flashcards - Regenerating for video_id: {video_id}")
        if video_id not in cache:
            return templates.TemplateResponse("error.html", {"request": request, "error_message": "Session expired or video not analyzed."})

        transcript_text = " ".join(cache[video_id]["chunks"])
        initial_flashcards = cache[video_id]["flashcards"]

# Later (e.g. on a “More” button click)
        more_flashcards = generate_additional_flashcards(
        transcript_text=transcript_text,
        existing_flashcards=initial_flashcards,
        num_cards=5
        )

        # Update cache
        combined_flashcards = initial_flashcards.strip() + "\n\n" + more_flashcards.strip()
        cache[video_id]["flashcards"] = combined_flashcards

        logger.info("Flashcards regenerated successfully.")
        return templates.TemplateResponse("result.html", {
            "request": request,
            "title": cache[video_id]["title"],
            "duration": cache[video_id]["duration"],
            "upload_date": cache[video_id]["upload_date"],
            "summary": cache[video_id]["summary"],
            "flashcards": combined_flashcards,
            "video_id": video_id
        })
    except Exception as e:
        logger.exception("Error in /flashcards route")
        return templates.TemplateResponse("error.html", {"request": request, "error_message": str(e)})
