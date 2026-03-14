import os
import shutil
import asyncio
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from frame_extractor import extract_frames
from describer import describe_all_frames
from indexer import index_frames
from query_engine import answer_query

load_dotenv()

app = FastAPI(title="Video RAG Query System")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global status tracker
processing_status = {
    "is_processing": False,
    "step": "",
    "progress": 0,
    "done": False,
    "error": "",
    "frames_extracted": 0,
    "frames_indexed": 0
}

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

async def process_video_background(video_path: str):
    """Runs video processing in background."""
    global processing_status
    try:
        processing_status.update({"is_processing": True, "step": "Extracting frames", "progress": 10, "done": False, "error": ""})
        frames = extract_frames(video_path, output_folder="frames", interval_seconds=2)
        if not frames:
            processing_status.update({"error": "Could not extract frames.", "is_processing": False})
            return

        processing_status.update({"step": "AI describing frames", "progress": 40, "frames_extracted": len(frames)})
        enriched_frames = describe_all_frames(frames)

        processing_status.update({"step": "Indexing to vector DB", "progress": 80})
        count = index_frames(enriched_frames)

        processing_status.update({
            "step": "Done!",
            "progress": 100,
            "done": True,
            "is_processing": False,
            "frames_indexed": count
        })
    except Exception as e:
        processing_status.update({"error": str(e), "is_processing": False, "progress": 0})

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    global processing_status

    if not file.filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Only video files are supported.")

    if processing_status["is_processing"]:
        raise HTTPException(status_code=400, detail="Already processing a video. Please wait.")

    # Save uploaded video
    contents = await file.read()
    video_path = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    with open(video_path, "wb") as f:
        f.write(contents)

    # Reset status
    processing_status.update({"done": False, "error": "", "progress": 0, "step": "Starting..."})

    # Start background processing
    asyncio.create_task(process_video_background(video_path))

    # Return immediately — don't wait
    return {"message": "Video upload successful! Processing started.", "filename": file.filename}

@app.get("/processing-status")
async def get_processing_status():
    """Frontend polls this to check progress."""
    return processing_status

@app.post("/query")
async def query_video(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    result = answer_query(request.query)
    return result

@app.get("/status")
async def get_status():
    chroma_path = "./chroma_db"
    has_data = os.path.exists(chroma_path) and len(os.listdir(chroma_path)) > 0
    return {
        "indexed": has_data,
        "message": "Video is indexed and ready to query!" if has_data else "No video indexed yet."
    }