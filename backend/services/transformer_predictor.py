
import librosa
import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_id = "soumili-04/audio-deepfake-approach2"

feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

model = AutoModelForAudioClassification.from_pretrained(
    model_id
).to(device)

model.eval()


def direct_predict(audio_path):
    arr, sr = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

    arr, _ = librosa.effects.trim(
        arr,
        top_db=25
    )

    inputs = feature_extractor(
        arr,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits

        probabilities = (
            F.softmax(logits, dim=-1)
            .squeeze()
            .cpu()
            .numpy()
        )

        pred_idx = torch.argmax(logits, dim=-1).item()

    label = model.config.id2label.get(
        pred_idx,
        f"Class {pred_idx}"
    )

    confidence = probabilities[pred_idx] * 100

    return label, float(confidence)
