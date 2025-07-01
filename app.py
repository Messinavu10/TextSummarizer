from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
import uvicorn
import sys
import os
from pathlib import Path
from src.textSummarizer.pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI(title="Text Summarizer", description="AI-powered text summarization tool")

# Create templates directory if it doesn't exist
templates_dir = Path("templates")
templates_dir.mkdir(exist_ok=True)

# Create static directory if it doesn't exist
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return {"status": "success", "message": "Training completed successfully!"}
    except Exception as e:
        return {"status": "error", "message": f"Error occurred during training: {str(e)}"}

@app.post("/predict")
async def predict_route(text: str = Form(...)):
    try:
        if not text.strip():
            return {"status": "error", "message": "Please enter some text to summarize"}
        
        obj = PredictionPipeline()
        summary = obj.predict(text)
        return {"status": "success", "summary": summary, "original_text": text}
    except Exception as e:
        return {"status": "error", "message": f"Error occurred during prediction: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
