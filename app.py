import os
import sys
import subprocess
import base64
import binascii

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

from cnnclassifier.pipeline.predict import PredictionPipeline

os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"


clApp = ClientApp()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
   return templates.TemplateResponse(request=request, name="index.html")


@app.post("/train")
async def train_route():
    subprocess.run([sys.executable, "main.py"], check=True)
    return {"message": "Training done successfully!"}

@app.post("/predict")
async def predict_route(request: Request):
    data = await request.json()
    image = data.get("image", "")

    if not image:
        raise HTTPException(status_code=400, detail="Image not provided")

    try:
        imgdata = base64.b64decode(image.split(",")[-1] + "==")
    except binascii.Error:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    with open(clApp.filename, "wb") as f:
        f.write(imgdata)

    prediction = PredictionPipeline(clApp.filename).predict()

    return JSONResponse(content={
        "class": str(prediction.get("class", prediction) if isinstance(prediction, dict) else prediction),
        "confidence": float(prediction.get("confidence", 1.0) if isinstance(prediction, dict) else 1.0),
    })