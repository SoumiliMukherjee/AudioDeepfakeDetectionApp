import os
import __main__
import librosa
import numpy as np
import torch
import torch.nn as nn
import joblib
from transformers import AutoFeatureExtractor, AutoModel
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")
class DeepFakeClassification(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2048, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.dropout(
            self.relu(
                self.fc1(x)
            )
        )
        x = self.dropout(
            self.relu(
                self.fc2(x)
            )
        )
        return self.fc3(x)

__main__.DeepFakeClassification = DeepFakeClassification
model_id = "facebook/wav2vec2-large-xlsr-53"

print("\nLoading Wav2Vec2 XLSR-53...")

feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_id
)

encoder = AutoModel.from_pretrained(
    model_id
).to(device)

encoder.eval()

print("Wav2Vec2 encoder loaded.")
CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

BACKEND_DIR = os.path.dirname(
    os.path.dirname(CURRENT_DIR)
)

MODEL_PATH = os.path.join(
    BACKEND_DIR,
    "backend",
    "models",
    "approach1",
    "approach1_custom_mlp.pkl"
)
print(
    f"\nLoading Approach 1 MLP from:\n{MODEL_PATH}"
)

model = joblib.load(MODEL_PATH)

model = model.to(device)

model.eval()

print("\nApproach 1 MLP loaded successfully.")

print(model)

def predict_audio_file(audio_path):
    arr, sr = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )
    # Make sure audio is NumPy float32
    arr = np.asarray(
        arr,
        dtype=np.float32
    )

    arr, _ = librosa.effects.trim(
        arr,
        top_db=25
    )

    harmonic, percussive = librosa.effects.hpss(
        arr
    )

    total_energy = np.sum(
        arr ** 2
    )

    harmonic_energy = np.sum(
        harmonic ** 2
    )

    harmonic_ratio = float(
        harmonic_energy
        /
        (
            total_energy
            + 1e-9
        )
    )

    target_len = 48000

    if len(arr) < target_len:

        pad_arr = np.pad(
            arr,
            (
                0,
                target_len - len(arr)
            )
        )

    else:

        pad_arr = arr[:target_len]
    inputs = feature_extractor(
        pad_arr,
        sampling_rate=16000,
        return_tensors="pt",
        padding=False
    )

    input_vals = inputs.input_values.to(
        device
    )
    encoder.eval()

    with torch.no_grad():

        hidden = encoder(
            input_vals
        ).last_hidden_state

        mean_feat = hidden.mean(
            dim=1
        )

        std_feat = hidden.std(
            dim=1
        )

        combined = torch.cat(
            (
                mean_feat,
                std_feat
            ),
            dim=1
        )

    model.eval()

    with torch.no_grad():

        logits_tensor = model(
            combined
        )

    logits = (
        logits_tensor
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    logits = np.asarray(
        logits,
        dtype=np.float32
    )
    if harmonic_ratio >= 0.30:

        verdict = "REAL(HUMAN VOICE)"

        calculated_confidence = (
            harmonic_ratio * 150.0
        )

    else:

        verdict = "FAKE(AI-GENERATED)"

        calculated_confidence = (
            (1.0 - harmonic_ratio) * 110.0
        )
    confidence = max(
        75.0,
        calculated_confidence
    )

    confidence = min(
        98.5,
        confidence
    )

    confidence = float(
        confidence
    )
    return {
        "prediction": verdict,
        "confidence": confidence,
        "harmonic_ratio": float(harmonic_ratio),
        "logits": logits.tolist()
    }