import streamlit as st
import whisper
import tempfile

# --- En-tête moderne ---
# --- En-tête style image ---
st.markdown(
    """
    <div style="
        background-color:black;
        color:white;
        padding:20px;
        text-align:center;
        font-size:28px;
        font-weight:bold;
        letter-spacing:2px;
    ">
        ✨ Created by Abdessamad Karim ✨
    </div>
    """,
    unsafe_allow_html=True
)

st.set_page_config(page_title="Transcription Audio", layout="centered")

st.title("🎙️ Transcription Audio avec Whisper")

@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

uploaded_file = st.file_uploader("Choisissez un fichier audio", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    # Sauvegarde temporaire
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_audio = tmp.name

    st.write("⏳ Transcription en cours...")
    result = model.transcribe(temp_audio, fp16=False)

    st.success("✅ Transcription terminée")
    st.subheader("Texte transcrit :")
    st.write(result["text"])

