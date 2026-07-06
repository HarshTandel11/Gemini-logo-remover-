import os
import uuid
import logging
import asyncio
import cv2
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import backend.config as Config
from backend.pipeline.orchestrator import PipelineOrchestrator
from typing import Optional, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("app")

app = FastAPI(title="Gemini Video Logo Remover Backend")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store job states in-memory
# format: { job_id: { "status": str, "progress": float, "stage": str, "message": str, "output_path": str } }
jobs = {}
gpu_lock = asyncio.Lock()

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    stage: str
    message: str

def run_pipeline_task(job_id: str, input_path: str, output_path: str, logo_bbox: Optional[Tuple[int, int, int, int]] = None):
    """
    Synchronous pipeline task executed in the background.
    """
    logger.info(f"Starting background job {job_id}...")
    orchestrator = PipelineOrchestrator()
    
    def progress_callback(stage: str, progress: float, message: str = ''):
        jobs[job_id]["progress"] = progress
        jobs[job_id]["stage"] = stage
        jobs[job_id]["message"] = message
        logger.info(f"Job {job_id} [{stage}] {progress:.0%} - {message}")

    orchestrator.set_progress_callback(progress_callback)
    
    try:
        jobs[job_id]["status"] = "processing"
        result_path = orchestrator.process(input_path, output_path, logo_bbox=logo_bbox)
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["stage"] = "done"
        jobs[job_id]["message"] = "Processing complete!"
        jobs[job_id]["output_path"] = result_path
        logger.info(f"Job {job_id} completed successfully.")
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = str(e)
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
    finally:
        # Clean up input file to save space
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except Exception as ex:
                logger.warning(f"Failed to delete input file {input_path}: {ex}")

async def run_pipeline_wrapper(job_id: str, input_path: str, output_path: str, logo_bbox: Optional[Tuple[int, int, int, int]] = None):
    """
    Wrapper to enforce single GPU task execution using an asyncio lock.
    """
    jobs[job_id]["status"] = "queued"
    jobs[job_id]["message"] = "Waiting in queue for GPU..."
    
    async with gpu_lock:
        # Run pipeline in a separate thread to avoid blocking FastAPI event loop
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, run_pipeline_task, job_id, input_path, output_path, logo_bbox)

@app.post("/api/upload")
async def upload_video(
    file: UploadFile = File(...),
    x: Optional[int] = None,
    y: Optional[int] = None,
    w: Optional[int] = None,
    h: Optional[int] = None,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Accepts video upload, generates a unique job ID, and spawns the background logo removal task.
    """
    # Verify file format
    ext = Path(file.filename).suffix.lower()
    if ext not in Config.SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats: {', '.join(Config.SUPPORTED_FORMATS)}"
        )
        
    job_id = str(uuid.uuid4())
    input_filename = f"input_{job_id}{ext}"
    output_filename = f"output_{job_id}.mp4"
    
    input_path = str(Config.UPLOAD_DIR / input_filename)
    output_path = str(Config.OUTPUT_DIR / output_filename)
    
    logo_bbox = None
    if x is not None and y is not None and w is not None and h is not None:
        logo_bbox = (x, y, w, h)
    
    logger.info(f"Receiving upload. Creating Job ID: {job_id}")
    
    try:
        # Save uploaded file
        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        logger.error(f"Failed to save upload: {e}")
        raise HTTPException(status_code=500, detail="Failed to write file to disk.")
        
    # Initialise job state
    jobs[job_id] = {
        "status": "received",
        "progress": 0.0,
        "stage": "upload",
        "message": "File uploaded, queued for processing.",
        "output_path": ""
    }
    
    # Run in background
    background_tasks.add_task(run_pipeline_wrapper, job_id, input_path, output_path, logo_bbox)
    
    return {"job_id": job_id, "status": "queued"}

@app.get("/api/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Retrieve status and progress of a specific background job.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    
    job = jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "stage": job["stage"],
        "message": job["message"]
    }

@app.get("/api/download/{job_id}")
async def download_output(job_id: str):
    """
    Download the final processed video file.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
        
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job is not completed yet.")
        
    out_path = job["output_path"]
    if not os.path.exists(out_path):
        raise HTTPException(status_code=404, detail="Output file not found on disk.")
        
    return FileResponse(
        out_path,
        media_type="video/mp4",
        filename=f"clean_video_{job_id[:8]}.mp4"
    )

@app.post("/api/extract")
async def extract_logo_endpoint(
    file: UploadFile = File(...),
    x: Optional[int] = None,
    y: Optional[int] = None,
    w: Optional[int] = None,
    h: Optional[int] = None,
):
    """
    Extract the specified (or auto-detected) logo region from the first frame of the video.
    """
    # Verify file format
    ext = Path(file.filename).suffix.lower()
    if ext not in Config.SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats: {', '.join(Config.SUPPORTED_FORMATS)}"
        )
        
    temp_in = Config.TEMP_DIR / f"temp_ext_{uuid.uuid4()}{ext}"
    try:
        # Save file to temp directory
        with open(temp_in, "wb") as f:
            content = await file.read()
            f.write(content)
            
        cap = cv2.VideoCapture(str(temp_in))
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise HTTPException(status_code=500, detail="Could not read video frame.")
            
        fh, fw = frame.shape[:2]
        
        if x is None or y is None or w is None or h is None:
            # Auto-detect logo region
            from backend.pipeline.stage2_sam2 import SAM2Segmenter
            
            # Create a dummy config class matching Config properties for SAM2Segmenter
            class DummyConfig:
                SAM2_CHECKPOINT = Config.SAM2_CHECKPOINT
            
            segmenter = SAM2Segmenter(DummyConfig(), None)
            bbox = segmenter.detect_logo_region(frame)
        else:
            bbox = (x, y, w, h)
            
        bx, by, bw, bh = bbox
        
        # Clamp to frame boundary
        bx = max(0, min(bx, fw - 1))
        by = max(0, min(by, fh - 1))
        bw = max(1, min(bw, fw - bx))
        bh = max(1, min(bh, fh - by))
        
        cropped = frame[by:by+bh, bx:bx+bw]
        
        temp_out = Config.TEMP_DIR / f"logo_{uuid.uuid4()}.png"
        cv2.imwrite(str(temp_out), cropped)
        
        return FileResponse(
            str(temp_out),
            media_type="image/png",
            filename="extracted_logo.png"
        )
    except Exception as e:
        logger.error(f"Error extracting logo: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
            try:
                temp_in.unlink()
            except Exception:
                pass

@app.post("/api/detect-logo")
async def detect_logo_endpoint(file: UploadFile = File(...)):
    """
    Auto-detect the logo region coordinates from the first frame of the video.
    """
    # Verify file format
    ext = Path(file.filename).suffix.lower()
    if ext not in Config.SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats: {', '.join(Config.SUPPORTED_FORMATS)}"
        )
        
    temp_in = Config.TEMP_DIR / f"temp_det_{uuid.uuid4()}{ext}"
    try:
        # Save file to temp directory
        with open(temp_in, "wb") as f:
            content = await file.read()
            f.write(content)
            
        cap = cv2.VideoCapture(str(temp_in))
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise HTTPException(status_code=500, detail="Could not read video frame.")
            
        # Auto-detect logo region
        from backend.pipeline.stage2_sam2 import SAM2Segmenter
        
        class DummyConfig:
            SAM2_CHECKPOINT = Config.SAM2_CHECKPOINT
        
        segmenter = SAM2Segmenter(DummyConfig(), None)
        bbox = segmenter.detect_logo_region(frame)
        
        return {"x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]}
    except Exception as e:
        logger.error(f"Error detecting logo coordinates: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_in.exists():
            try:
                temp_in.unlink()
            except Exception:
                pass

if __name__ == "__main__":
    import uvicorn
    # Make sure app runs on 0.0.0.0:8000 or config-defined port
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
