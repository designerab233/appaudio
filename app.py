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

st.title("🎙️ Transcription Audio avec Whisper")

# ------------------ Utilitaires ------------------
def has_ffmpeg() -> bool:
    """Vérifie si ffmpeg est disponible dans le PATH."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

def convert_to_wav(input_path: str, output_path: str) -> None:
    """Convertit un fichier audio en WAV 16-bit 16k/44.1k via ffmpeg."""
    # Utilise ffmpeg pour convertir en wav (PCM S16LE)
    cmd = [
        "ffmpeg",
        "-y",  # overwrite
        "-i", input_path,
        "-ar", "16000",  # sample rate 16 kHz (bon pour la transcription)
        "-ac", "1",      # mono
        "-sample_fmt", "s16",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

# ------------------ Chargement du modèle ------------------
@st.cache_resource
def load_model(name: str = "base"):
    """Charge et met en cache le modèle Whisper choisi."""
    try:
        model = whisper.load_model(name)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None

# Choix du modèle
st.sidebar.markdown("### ⚙️ Paramètres du modèle")
model_choice = st.sidebar.selectbox("Taille du modèle Whisper", ["tiny", "base", "small", "medium", "large"], index=1)
use_fp16 = st.sidebar.checkbox("Utiliser fp16 (si GPU compatible)", value=False)

model = load_model(model_choice)
if model is None:
    st.stop()

# ------------------ Upload du fichier ------------------
uploaded_file = st.file_uploader("Choisissez un fichier audio", type=["mp3", "wav", "m4a", "flac", "ogg", "aac"])
if uploaded_file is None:
    st.info("Importez un fichier audio (mp3, wav, m4a, flac, ogg, aac).")
    st.stop()

# Affichage du player audio
st.audio(uploaded_file)

# Création d'un fichier temporaire avec le bon suffixe
original_filename = Path(uploaded_file.name).stem
suffix = Path(uploaded_file.name).suffix or ".wav"

with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_orig:
    uploaded_file.seek(0)
    tmp_orig.write(uploaded_file.read())
    tmp_orig_path = tmp_orig.name

# Préparation du fichier WAV pour Whisper (si nécessaire)
wav_path = tmp_orig_path
converted = False
if Path(tmp_orig_path).suffix.lower() != ".wav":
    if has_ffmpeg():
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                wav_path = tmp_wav.name
            convert_to_wav(tmp_orig_path, wav_path)
            converted = True
        except subprocess.CalledProcessError:
            st.warning("La conversion via ffmpeg a échoué. Whisper peut parfois accepter le fichier original, on va essayer.")
            wav_path = tmp_orig_path
    else:
        st.warning("ffmpeg n'est pas trouvé sur le système. Whisper prendra le fichier original si possible, sinon installez ffmpeg pour une meilleure compatibilité.")
        wav_path = tmp_orig_path

# Bouton pour lancer la transcription
if st.button("⏳ Lancer la transcription"):
    try:
        st.info("Transcription en cours...")
        # whisper.transcribe accepte souvent différents formats si ffmpeg est installé.
        # On force fp16 à False si l'utilisateur l'a désactivé ou si le modèle large sur CPU provoque des erreurs.
        result = model.transcribe(wav_path, fp16=use_fp16)
        text = result.get("text", "").strip()

        if not text:
            st.warning("Aucune transcription récupérée (texte vide). Vérifiez le fichier ou essayez un autre modèle.")
        else:
            st.success("✅ Transcription terminée")
            st.subheader("Texte transcrit :")
            st.write(text)

            # Bouton pour télécharger la transcription en .txt
            txt_bytes = text.encode("utf-8")
            st.download_button(
                label="⬇️ Télécharger la transcription (.txt)",
                data=txt_bytes,
                file_name=f"{original_filename}_transcription.txt",
                mime="text/plain",
            )

            # Optionnel : sauvegarder en local (dans dossier courant) pour debugging / commit
            save_locally = st.checkbox("Sauvegarder une copie locale du fichier .txt (serveur)", value=False)
            if save_locally:
                out_path = Path(f"{original_filename}_transcription.txt").resolve()
                out_path.write_text(text, encoding="utf-8")
                st.write(f"Copie sauvegardée sur le serveur : {out_path}")

    except Exception as e:
        st.error(f"Une erreur est survenue pendant la transcription : {e}")

# Nettoyage optionnel des fichiers temporaires (on laisse pour debug, mais tu peux supprimer)
st.caption("Note: des fichiers temporaires sont créés sur le serveur pendant la conversion. Ils sont non supprimés pour faciliter le debug.")
