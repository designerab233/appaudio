# app.py
import os
import tempfile
import subprocess
from pathlib import Path

import streamlit as st
import whisper

# ------------------ Configuration de la page ------------------
st.set_page_config(page_title="Transcription Audio", layout="centered")
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
    unsafe_allow_html=True,
)

st.title("🎙️ Transcription Audio (Anglais) avec Whisper")

# ------------------ Fonctions utilitaires ------------------
def has_ffmpeg() -> bool:
    """Vérifie si ffmpeg est installé."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

def convert_to_wav(input_path: str, output_path: str) -> None:
    """Convertit un fichier audio en WAV (mono, 16kHz)."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-sample_fmt", "s16",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

# ------------------ Chargement du modèle ------------------
@st.cache_resource
def load_model(name: str = "base"):
    """Charge le modèle Whisper."""
    try:
        model = whisper.load_model(name)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

# ------------------ Paramètres ------------------
st.sidebar.markdown("### ⚙️ Paramètres du modèle")
model_choice = st.sidebar.selectbox("Taille du modèle Whisper", ["tiny", "base", "small", "medium", "large"], index=1)
use_fp16 = st.sidebar.checkbox("Utiliser fp16 (si GPU disponible)", value=False)

model = load_model(model_choice)
if model is None:
    st.stop()

# ------------------ Upload du fichier ------------------
uploaded_file = st.file_uploader("Choisissez un fichier audio", type=["mp3", "wav", "m4a", "flac", "ogg", "aac"])
if uploaded_file is None:
    st.info("Importez un fichier audio en anglais (mp3, wav, m4a, flac, ogg, aac).")
    st.stop()

# Lecture de l'audio
st.audio(uploaded_file)

# Sauvegarde temporaire
original_filename = Path(uploaded_file.name).stem
suffix = Path(uploaded_file.name).suffix or ".wav"

with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_orig:
    uploaded_file.seek(0)
    tmp_orig.write(uploaded_file.read())
    tmp_orig_path = tmp_orig.name

# Conversion en .wav si nécessaire
wav_path = tmp_orig_path
if Path(tmp_orig_path).suffix.lower() != ".wav" and has_ffmpeg():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        wav_path = tmp_wav.name
    try:
        convert_to_wav(tmp_orig_path, wav_path)
    except subprocess.CalledProcessError:
        st.warning("⚠️ Échec de la conversion via ffmpeg. On utilisera le fichier original.")
        wav_path = tmp_orig_path

# ------------------ Transcription ------------------
if st.button("⏳ Lancer la transcription"):
    try:
        st.info("Transcription en cours... veuillez patienter...")
        # Force la langue anglaise pour de meilleurs résultats
        result = model.transcribe(wav_path, fp16=use_fp16, language="en")
        text = result.get("text", "").strip()

        if not text:
            st.warning("Aucun texte détecté. Vérifiez le fichier audio ou essayez un autre modèle.")
        else:
            st.success("✅ Transcription terminée avec succès !")
            st.subheader("📝 Texte transcrit :")
            st.write(text)

            # Bouton de téléchargement
            st.download_button(
                label="⬇️ Télécharger la transcription (.txt)",
                data=text.encode("utf-8"),
                file_name=f"{original_filename}_transcription.txt",
                mime="text/plain",
            )

    except Exception as e:
        st.error(f"❌ Une erreur est survenue pendant la transcription : {e}")

st.caption("💡 Conseil : ce modèle est optimisé pour la langue anglaise.")
