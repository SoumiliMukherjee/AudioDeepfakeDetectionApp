import os
import sys
import librosa
import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor , AutoModelForAudioClassification

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR =  os.path.dirname(CURRENT_DIR)
MODEL_PATH = os.path.join(
    BACKEND_DIR,
    'models',
    'approach2_model'
)
print(f"Loading model from:\n{MODEL_PATH}")
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
model = AutoModelForAudioClassification.from_pretrained(MODEL_PATH).to(device)

model.eval()
print(model.config.id2label)

def predict_audio(audio_path):
    audio , sr = librosa.load(audio_path , sr=16000 , mono=True)
    audio , _ = librosa.effects.trim(audio , top_db=25)
    print(f"Sample rate:{sr}")
    print(f"Audio sample:{len(audio)}")

    inputs = feature_extractor(audio , sampling_rate = 16000 , return_tensors = 'pt' , padding = True)
    input_vals = inputs.input_values.to(device)

    with torch.no_grad():
        output = model(input_values=input_vals)
        logits = output.logits
        probabilities = F.softmax(logits , dim = -1).squeeze().cpu().numpy()
        pred_idx = torch.argmax(logits , dim = -1).item()
        label = model.config.id2label.get(pred_idx,f"Class {pred_idx}")
        confidence = float(probabilities[pred_idx] * 100)
    print(f"Raw logits: {logits.squeeze().cpu().numpy()}")
    print(f"Probabilities: {probabilities}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2f}%")

    return {
        "prediction": label,
        "confidence": confidence,
        "probabilities": probabilities.tolist()
    }

if __name__ == "__main__":

    TEST_AUDIO = os.path.join(
    BACKEND_DIR,
    "..",
    "sample_audio",
    "india_current_affairs_ai_voice.wav"
)

    result = predict_audio(TEST_AUDIO)

    print(result)


