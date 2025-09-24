"""
FastAPI Model API for Handwritten Digit Recognition
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import joblib
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Digit Recognition API", version="1.0.0")

model_path = "./models/digits_model.pkl"

try:
    model = joblib.load(model_path)
    print("Digits model loaded successfully!")
except FileNotFoundError as e:
    print(f"Model file not found: {e}")
    print("Please run the training script first.")
    model = None


class Pixels(BaseModel):
    pixels: list


class DigitPrediction(BaseModel):
    prediction: int
    confidence: float


def preprocess_image_to_8x8(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize((8, 8))
    arr = np.array(image, dtype=np.float32)
    arr = (arr / 255.0) * 16.0
    return arr.reshape(1, -1)


@app.get("/")
async def root():
    return {"message": "Digit Recognition API", "status": "running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict_pixels", response_model=DigitPrediction)
async def predict_from_pixels(payload: Pixels):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        pixels = np.array(payload.pixels, dtype=np.float32)
        if pixels.size != 64:
            raise ValueError("pixels must contain exactly 64 values for 8x8 image")
        if pixels.max() > 16.0:
            pixels = (pixels / 255.0) * 16.0
        features = pixels.reshape(1, -1)
        pred = int(model.predict(features)[0])
        probs = model.predict_proba(features)[0]
        conf = float(np.max(probs))
        return DigitPrediction(prediction=pred, confidence=conf)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.post("/predict_image", response_model=DigitPrediction)
async def predict_from_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        content = await file.read()
        features = preprocess_image_to_8x8(content)
        pred = int(model.predict(features)[0])
        probs = model.predict_proba(features)[0]
        conf = float(np.max(probs))
        return DigitPrediction(prediction=pred, confidence=conf)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
