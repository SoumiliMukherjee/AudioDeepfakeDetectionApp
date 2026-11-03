
import streamlit as st
import tempfile
import os
from gradio_client import Client, handle_file


st.set_page_config(
    page_title="VoiceGuard AI",
    page_icon="🎙️",
    layout="wide"
)


# Connect to Hugging Face Space
@st.cache_resource
def get_hf_client():
    return Client("soumili-04/voiceguard-ai")


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


# Main heading
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


# Model selection
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


# Maintain selected model
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "approach1"


if vocalguard_selected:
    st.session_state.selected_model = "approach1"


if deepvoice_selected:
    st.session_state.selected_model = "approach2"


st.divider()


# Display selected model
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


# Audio input section
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


# Upload audio
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


# Record audio
else:

    recorded_audio = st.audio_input(
        "Click the microphone button and record your voice"
    )

    if recorded_audio is not None:
        uploaded_file = recorded_audio


# Audio preview and analysis
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

        temp_path = None

        with st.spinner(
            "Analyzing audio... This may take a moment."
        ):

            try:

                # Save audio temporarily
                suffix = os.path.splitext(
                    uploaded_file.name
                )[1] or ".wav"

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getvalue()
                    )

                    temp_path = temp_file.name


                # Convert selected model
                # to Hugging Face API format
                approach = (
                    "Approach 1"
                    if st.session_state.selected_model == "approach1"
                    else "Approach 2"
                )


                # Connect to Hugging Face Space
                client = get_hf_client()


                # Call Gradio API
                result = client.predict(
                    audio_path=handle_file(temp_path),
                    approach=approach,
                    api_name="/detect_audio"
                )


                # Extract API outputs
                prediction_text = result[0]
                model_details = result[1]


                # Display results
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

                st.subheader("Prediction")

                st.markdown(
                    prediction_text
                )

                st.subheader("Model Details")

                st.markdown(
                    model_details
                )


            except Exception as e:

                st.error(
                    f"Audio analysis failed: {str(e)}"
                )


            finally:

                # Delete temporary audio file
                if (
                    temp_path is not None
                    and os.path.exists(temp_path)
                ):

                    os.remove(temp_path)


# Footer
st.divider()

st.caption(
    "VoiceGuard AI • Audio Deepfake Detection System"
)
