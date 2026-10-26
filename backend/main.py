import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from backend.services.hybrid_predictor import predict_audio_file
from backend.services.transformer_predictor import direct_predict

app = FastAPI(
    title="Audio Deepfake Detection API",
    description="Voice Fraud Detection using two detection approaches",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_AUDIO_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "temp_audio")
)
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
@app.get("/")
def root():
    return {
        "message": "Audio Deepfake Detection API is running",
        "status": "active",
        "available_models": [
            "approach1",
            "approach2"
        ]
    }
@app.post("/predict")
async def predict(   
    file: UploadFile = File(...),
    approach: str = Form(...)
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file provided"
        )
    allowed_extensions = {
        ".wav",
        ".mp3",
        ".ogg",
        ".flac",
        ".m4a"
    }
    extension = os.path.splitext(
        file.filename 
    )   [1].lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported audio format. "
                "Use WAV, MP3, OGG, FLAC or M4A."
            )
        )
    if approach not in ["approach1", "approach2"]:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid approach. "
                "Choose approach1 or approach2."
            )
        )
    temp_path = os.path.join(
        TEMP_AUDIO_DIR,
        file.filename
    )
    try:
        contents = await file.read()
        with open(temp_path, "wb") as audio_file:
            audio_file.write(contents)
        if approach == "approach1":

            result = predict_audio_file(
                temp_path
            )
            result["approach"] = (
                "Approach 1 - "
                "Custom MLP + Harmonic Analysis"
            )
        elif approach == "approach2":

            prediction, confidence = direct_predict(
                temp_path
            )
            result = {
                "prediction": prediction,
                "confidence": float(confidence),
                "approach": (
                    "Approach 2 - "
                    "MelodyMachine Deepfake Audio Detection V2"
                )
            }
        result["filename"] = file.filename
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)