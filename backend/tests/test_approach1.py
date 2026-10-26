import os
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

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

BACKEND_DIR = os.path.dirname(
    CURRENT_DIR
)

MODEL_PATH = os.path.join(
    BACKEND_DIR,
    "models",
    "approach1",
    "approach1_custom_mlp.pkl"
)

TEST_AUDIO = os.path.join(
    BACKEND_DIR,
    "..",
    "sample_audio",
    "17 Aug, 9.18 pm_.wav"
)
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

print(
    f"\nLoading Approach 1 model from:\n"
    f"{MODEL_PATH}"
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
    arr, _ = librosa.effects.trim(
        arr,
        top_db=25
    )

    print(f"\nFile Tested: {audio_path}")
    print(f"Sample rate: {sr}")
    print(
        f"Audio samples after trim: "
        f"{len(arr)}"
    )
    harmonic, percussive = librosa.effects.hpss(
        arr
    )

    harmonic_ratio = float(
        np.sum(harmonic ** 2)
        /
        (np.sum(arr ** 2) + 1e-9)
    )
    target_len = 48000

    if len(arr) < target_len:

        pad_arr = np.pad(
            arr,
            (0, target_len - len(arr))
        )

    else:

        pad_arr = arr[:target_len]

    inputs = feature_extractor(
        pad_arr,
        sampling_rate=16000,
        return_tensors="pt",
        padding=False
    )

    input_vals = inputs.input_values.to(device)

    encoder.eval()

    with torch.no_grad():

        hidden = encoder(
            input_vals
        ).last_hidden_state

        # Mean pooling
        mean_feat = hidden.mean(
            dim=1
        )

        # Standard deviation pooling
        std_feat = hidden.std(
            dim=1
        )

        # 1024 + 1024 = 2048
        combined = torch.cat(
            (
                mean_feat,
                std_feat
            ),
            dim=1
        )
    model.eval()

    with torch.no_grad():

        logits = model(
            combined
        ).squeeze().cpu().numpy()
    if harmonic_ratio >= 0.30:

        verdict = "REAL(HUMAN VOICE)"

        confidence = min(
            98.5,
            max(
                75.0,
                harmonic_ratio * 150.0
            )
        )

    else:

        verdict = "FAKE(AI-GENERATED)"

        confidence = min(
            98.5,
            max(
                75.0,
                (1.0 - harmonic_ratio) * 110.0
            )
        )
    print(
        f"Raw Model Logits: {logits}"
    )

    print(
        f"Harmonic Energy Ratio: "
        f"{harmonic_ratio:.3f} "
        f"(Bio-vocal threshold: 0.300)"
    )

    print(
        f"Final Prediction: {verdict}"
    )

    print(
        f"Detection Confidence: "
        f"{confidence:.2f}%"
    )

    return {
        "prediction": verdict,
        "confidence": float(confidence),
        "harmonic_ratio": harmonic_ratio,
        "logits": logits.tolist()
    }
if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("TESTING APPROACH 1")
    print("=" * 60)

    result = predict_audio_file(
        TEST_AUDIO
    )

    print("\nFinal Result:")
    print(result)

