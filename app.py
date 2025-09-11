import streamlit as st
import whisper
import tempfile

# --- En-tête moderne ---
st.markdown(
    """
    <div style="
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        margin-bottom: 30px;
        font-family: 'Inter', sans-serif;
    ">
        <div style="
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
        ">
            <h1 style="
                font-size: 4vw;
                font-weight: 900;
                background-color: #000;
                color: #fff;
                padding: 0.5em;
                margin: 0;
            ">Created by Abdessamad Karim</h1>
            <h1 style="
                position: absolute;
                font-size: 4vw;
                font-weight: 900;
                background-color: #fff;
                color: #000;
                margin: 0;
                padding: 0.5em;
                clip-path: inset(-1% -1% 50% -1%);
            ">Created by Abdessamad Karim</h1>
        </div>
        <p style="
            font-size: 1.5vw;
            font-weight: 900;
            margin-top: 1em;
            text-align: center;
        ">THANK YOU <span style='display:block; transform: rotate(90deg); margin-top:0.25em;'>:)</span></p>
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
