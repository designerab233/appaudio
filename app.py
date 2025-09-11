import streamlit as st
import whisper
import tempfile

# --- Configuration de la page ---
st.set_page_config(page_title="Transcription Audio", layout="centered")

# --- En-tête moderne inversé ---
st.markdown(
    """
    <div style="
        background-color:#222222;
        color:#ffffff;
        padding:25px;
        text-align:center;
        font-size:28px;
        font-weight:bold;
        letter-spacing:2px;
        border-radius:15px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        margin-bottom:30px;
        font-family: 'Inter', sans-serif;
    ">
        ✨ Created by Abdessamad Karim ✨
    </div>
    """,
    unsafe_allow_html=True
)

st.title("🎙️ Transcription Audio avec Whisper")

# --- Chargement modèle Whisper ---
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# --- Upload fichier audio ---
uploaded_file = st.file_uploader(
    "Choisissez un fichier audio",
    type=["mp3", "wav", "m4a"]
)

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
    st.markdown(
        f"<div style='background-color:#f5f5f5; padding:20px; border-radius:10px; font-family:Inter; white-space:pre-wrap; word-wrap:break-word;'>{result['text']}</div>",
        unsafe_allow_html=True
    )
