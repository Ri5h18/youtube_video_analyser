
@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, video_url: str = Form(...)):
    transcript = get_transcript(video_url)
    if not transcript:
        return templates.TemplateResponse("index.ejs", {"request": request, "error": "Transcript not found"})

    summary, flashcards, vector_index = process_text(transcript)
    return templates.TemplateResponse("result.ejs", {
        "request": request,
        "summary": summary,