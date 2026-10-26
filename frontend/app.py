import streamlit as st
import requests
st.set_page_config(
    page_title="VoiceGuard AI",
    page_icon="🎙️",
    layout="wide"
)
st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #777;
        margin-bottom: 35px;
    }

    .model-card {
        padding: 22px;
        border-radius: 15px;
        border: 1px solid #ddd;
        margin-bottom: 15px;
    }

    .model-title {
        font-size: 23px;
        font-weight: 650;
    }

    .model-description {
        font-size: 15px;
        color: #666;
        line-height: 1.6;
    }

    .result-title {
        font-size: 28px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    '<div class="main-title">🎙️ VoiceGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered voice authenticity and deepfake detection'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

st.subheader("🧠 Choose Detection Model")

st.write(
    "Select the detection engine you want to use for your audio."
)

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        """
        <div class="model-card">

        <div class="model-title">
        🛡️ VocalGuard
        </div>

        <br>

        <b>Hybrid Acoustic Intelligence</b>

        <p class="model-description">
        Combines Wav2Vec2-based neural audio embeddings
        with a custom neural classifier and harmonic
        voice analysis.
        </p>

        <p>
        ✓ Deep audio representations<br>
        ✓ Custom neural classifier<br>
        ✓ Harmonic acoustic analysis
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    vocalguard_selected = st.button(
        "Select VocalGuard",
        use_container_width=True
    )
with col2:
    st.markdown(
        """
        <div class="model-card">    

        <div class="model-title">
        🧠 DeepVoice AI
        </div>
        <br>
        <b>Transformer-Based Detection</b>
        <p class="model-description">
        Uses a pre-trained audio transformer to directly
        analyze speech and identify AI-generated or
        manipulated audio.
        </p>
        <p>
        ✓ Audio transformer<br>
        ✓ End-to-end classification<br>
        ✓ Direct deepfake prediction
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    deepvoice_selected = st.button(
        "Select DeepVoice AI",
        use_container_width=True
    )

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "approach1"

if vocalguard_selected:
    st.session_state.selected_model = "approach1"

if deepvoice_selected:
    st.session_state.selected_model = "approach2"
st.divider()

if st.session_state.selected_model == "approach1":

    st.success(
        "🛡️ **VocalGuard selected** — "
        "Hybrid acoustic + neural detection"
    )

else:

    st.success(
        "🧠 **DeepVoice AI selected** — "
        "Transformer-based deepfake detection"
    )
st.subheader("🎵 Provide Audio")

input_mode = st.radio(
    "Choose how you want to provide the audio:",
    [
        "📁 Upload Audio",
        "🎤 Record Voice"
    ],
    horizontal=True
)


uploaded_file = None

if input_mode == "📁 Upload Audio":

    uploaded_file = st.file_uploader(
        "Upload a voice recording for analysis",
        type=[
            "wav",
            "mp3",
            "ogg",
            "flac",
            "m4a"
        ]
    )

else:

    recorded_audio = st.audio_input(
        "Click the microphone button and record your voice"
    )

    if recorded_audio is not None:

        uploaded_file = recorded_audio
if uploaded_file is not None:

    st.divider()

    st.write("### 🎧 Audio Preview")

    st.audio(uploaded_file)

    st.write(
        f"**File:** `{uploaded_file.name}`"
    )

    st.divider()
    if st.button(
        "🔍 Analyze Voice",
        use_container_width=True
    ):

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }
        data = {
            "approach": st.session_state.selected_model
        }

        with st.spinner(
            "Analyzing audio... This may take a moment."
        ):

            try:

                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    files=files,
                    data=data
                )

                if response.status_code == 200:

                    result = response.json()

                    st.success(
                        "Analysis completed successfully."
                    )

                    st.divider()

                    st.markdown(
                        '<div class="result-title">'
                        '🔎 Detection Result'
                        '</div>',
                        unsafe_allow_html=True
                    )
                    st.write(
                        f"**Model:** "
                        f"{result.get('approach', 'Unknown')}"
                    )

                    prediction = result.get(
                        "prediction",
                        "Unknown"
                    )

                    confidence = result.get(
                        "confidence",
                        0
                    )
                    result_col1, result_col2 = st.columns(2)


                    with result_col1:

                        st.metric(
                            "Prediction",
                            prediction
                        )


                    with result_col2:

                        st.metric(
                            "Confidence",
                            f"{confidence:.2f}%"
                        )

                    if (
                        st.session_state.selected_model
                        == "approach1"
                    ):

                        harmonic_ratio = result.get(
                            "harmonic_ratio"
                        )

                        if harmonic_ratio is not None:

                            st.write(
                                f"**Harmonic Energy Ratio:** "
                                f"{harmonic_ratio:.3f}"
                            )

                else:
                    st.error(
                        f"Backend error: {response.text}"
                    )
            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Could not connect to the FastAPI backend."
                )

                st.info(
                    "Make sure FastAPI is running on "
                    "http://127.0.0.1:8000"
                )
st.divider()

st.caption(
    "VoiceGuard AI • Audio Deepfake Detection System"
)