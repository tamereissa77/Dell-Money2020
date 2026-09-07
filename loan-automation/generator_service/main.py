from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import os
import threading
from hr_factory import build_factory


app = FastAPI(title="HR Document Generator Service")

class GenerateRequest(BaseModel):
    count: int = 10
    output_dir: str = "/shared_data"

@app.post("/generate")
def generate_documents(req: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Endpoint to trigger the generation of synthetic HR documents.
    Generates images in the background to avoid blocking the API response.
    """
    os.makedirs(req.output_dir, exist_ok=True)
    
    # Run the generation in the background
    background_tasks.add_task(build_factory, req.count, req.output_dir)
    
    return {"status": "Generating", "count": req.count, "output_dir": req.output_dir}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
