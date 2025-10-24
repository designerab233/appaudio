import streamlit as st
import whisper
import tempfile
from pathlib import Path
import datetime
import io

# CONFIG (doit être appelé avant les autres éléments Streamlit visibles)
st.set_page_config(page_title="Transcription Audio", layout="centered")

# --- En-tête style inversé ---
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
        border-radius:8px;
    ">
        ✨ Created by Abdessamad Karim ✨
    </div>
    """,
    unsafe_allow_html=True
)

st.title("🎙️ Transcription Audio avec Whisper")

# Choix du modèle (tu peux proposer d'autres tailles : tiny, base, small, medium, large)
model_size = st.selectbox("Choisir la taille du modèle (impacte précision + vitesse)", ["base", "small", "medium"], index=0)

use_timestamps = st.checkbox("Inclure timestamps (segments)", value=False)
cpu_mode = st.checkbox("Forcer CPU (fp16 désactivé) — utile si pas de GPU", value=True)

@st.cache_resource
def load_model(size: str):
    """Charge et met en cache le modèle Whisper."""
    return whisper.load_model(size)

with st.spinner("Chargement du modèle..."):
    model = load_model(model_size)

uploaded_file = st.file_uploader("Choisissez un fichier audio (mp3, wav, m4a, etc.)", type=["mp3", "wav", "m4a", "ogg", "flac"])

if uploaded_file is not None:
    # Affiche le lecteur audio
    try:
        st.audio(uploaded_file)
    except Exception:
        # si st.audio échoue, on ignore (rare)
        pass

    # Détermine l'extension du fichier uploadé (sécurisée)
    original_name = getattr(uploaded_file, "name", None) or "audio"
    ext = Path(original_name).suffix if Path(original_name).suffix else ".wav"

    # Écrit le fichier uploadé dans un fichier temporaire (Whisper / ffmpeg lira ce fichier)
    uploaded_file.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.read())
        temp_audio_path = tmp.name

    st.info("⏳ Transcription en cours...")
    try:
        with st.spinner("Transcription en cours (cela peut prendre quelques secondes/minutes selon la taille du modèle et du fichier)..."):
            # Si CPU forcé, désactiver fp16
            result = model.transcribe(temp_audio_path, fp16=(not cpu_mode and model.device.type == "cuda"))
    except Exception as e:
        st.error(f"Une erreur est survenue lors de la transcription : {e}")
    else:
        st.success("✅ Transcription terminée")

        # Affiche le texte transcrit
        st.subheader("Texte transcrit :")
        transcript_text = result.get("text", "").strip()
        st.text_area("Transcription", value=transcript_text, height=300)

        # Si l'utilisateur veut les segments/timestamps, on crée un texte plus détaillé
        if use_timestamps:
            segments = result.get("segments", [])
            seg_lines = []
            for seg in segments:
                # format HH:MM:SS.mmm
                start = str(datetime.timedelta(seconds=int(seg["start"]))) + f"{seg['start']%1:.3f}".replace("0.", ".")
                end = str(datetime.timedelta(seconds=int(seg["end"]))) + f"{seg['end']%1:.3f}".replace("0.", ".")
                # Simplifier l'affichage des timestamps
                seg_lines.append(f"[{seg['start']:.2f}s → {seg['end']:.2f}s] {seg['text'].strip()}")
            segments_text = "\n".join(seg_lines)
            st.subheader("Segments (avec timestamps) :")
            st.text_area("Segments", value=segments_text, height=300)

        # Prépare un fichier téléchargeable (.txt)
        download_name = f"transcription_{Path(original_name).stem}.txt"
        out_buf = io.StringIO()
        out_buf.write(transcript_text + "\n")
        if use_timestamps:
            out_buf.write("\n\n--- Segments ---\n")
            out_buf.write(segments_text)
        out_buf.seek(0)
        st.download_button("⬇️ Télécharger la transcription", data=out_buf.getvalue(), file_name=download_name, mime="text/plain")

        # Info supplémentaire (optionnel)
        st.info(f"Modèle utilisé : {model_size} — Taille approximative du texte : {len(transcript_text.split())} mots")

        # Nettoyage (optionnel) : on peut supprimer le fichier temporaire si on veut,
        # mais ici on laisse le système s'en occuper ou l'OS le nettoiera.
