import os
import shutil
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

# Serve frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video, extract frames, describe them, and index into ChromaDB."""
    
    # Validate file type
    if not file.filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Only video files are supported.")

    # Save uploaded video
    video_path = f"uploads/{file.filename}"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(f"Video saved to {video_path}")

    # Step 1: Extract frames
    print("\n--- STEP 1: Extracting frames ---")
    frames = extract_frames(video_path, output_folder="frames", interval_seconds=2)

    if not frames:
        raise HTTPException(status_code=400, detail="Could not extract frames from video.")

    # Step 2: Describe frames with AI vision
    print("\n--- STEP 2: Describing frames with AI ---")
    enriched_frames = describe_all_frames(frames)

    # Step 3: Index into ChromaDB
    print("\n--- STEP 3: Indexing into vector store ---")
    count = index_frames(enriched_frames)

    return {
        "message": "Video processed successfully!",
        "filename": file.filename,
        "frames_extracted": len(frames),
        "frames_indexed": count
    }

@app.post("/query")
async def query_video(request: QueryRequest):
    """Query the indexed video using natural language."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    result = answer_query(request.query)
    return result

@app.get("/status")
async def get_status():
    """Check if a video has been indexed."""
    chroma_path = "./chroma_db"
    has_data = os.path.exists(chroma_path) and len(os.listdir(chroma_path)) > 0
    return {
        "indexed": has_data,
        "message": "Video is indexed and ready to query!" if has_data else "No video indexed yet."
    }