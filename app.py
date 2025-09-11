import streamlit as st
import whisper
import tempfile

# --- Configuration de la page ---
st.set_page_config(
    page_title="🎙️ Transcription Audio",
    page_icon="🎧",
    layout="centered"
)

# --- CSS global pour styliser l'app ---
st.markdown(
    """
    <style>
    /* Background gradient élégant */
    .main {
        background: linear-gradient(120deg, #fdfbfb 0%, #ebedee 100%);
        color: #333333;
        font-family: 'Inter', sans-serif;
    }

    /* Carte centrale pour le contenu */
    .stApp {
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .content-box {
        background-color: #ffffff;
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        max-width: 700px;
        width: 90%;
        text-align: center;
    }

    h1, h2, h3, h4 {
        font-weight: 800;
        color: #222222;
    }

    h1 {
        font-size: 2.5em;
        margin-bottom: 10px;
    }

    p {
        font-size: 1.1em;
        color: #555555;
    }

    .upload-btn {
        background-color: #4a90e2 !important;
        color: white !important;
        padding: 0.6em 1.2em;
        border-radius: 10px;
        font-weight: bold;
    }

    .upload-btn:hover {
        background-color: #357ABD !important;
    }

    .transcription-box {
        margin-top: 20px;
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 15px;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.05);
        text-align: left;
        white-space: pre-wrap;
        word-wrap: break-word;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# --- Header moderne avec effet superposé ---
st.markdown(
    """
    <div style="position: relative; display: flex; flex-direction: column; align-items: center; margin-bottom: 30px;">
        <div style="position: relative; display: flex; align-items: center; justify-content: center;">
            <h1 style="
                font-size: 3em;
                font-weight: 900;
                background-color: #222222;
                color: #fff;
                padding: 0.5em 1em;
                margin: 0;
            ">Abdessamad Karim</h1>
            <h1 style="
                position: absolute;
                font-size: 3em;
                font-weight: 900;
                background-color: #fff;
                color: #222222;
                margin: 0;
                padding: 0.5em 1em;
                clip-path: inset(-1% -1% 50% -1%);
            ">Abdessamad Karim</h1>
        </div>
        <p style="font-size:1.2em; font-weight: 700; margin-top: 10px; color:#555;">🎧 Transcription Audio Moderne</p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Box principale ---
st.markdown('<div class="content-box">', unsafe_allow_html=True)

st.write("**Téléchargez votre fichier audio et laissez Whisper transcrire pour vous.**")

# --- Chargement modèle Whisper ---
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

uploaded_file = st.file_uploader(
    label="Choisissez un fichier audio",
    type=["mp3", "wav", "m4a"],
    label_visibility="collapsed"
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
    st.markdown(f'<div class="transcription-box">{result["text"]}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
