import streamlit as st
import ffmpeg
import tempfile
import os
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import FreqDist, pos_tag
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
import heapq
import re
from faster_whisper import WhisperModel
from googletrans import Translator
from wordcloud import WordCloud
from langdetect import detect
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Setup NLTK
def setup_nltk():
    nltk_dependencies = {
        'stopwords': 'corpora/stopwords',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
    }
    for package, path in nltk_dependencies.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package)

setup_nltk()

# Streamlit config
st.set_page_config(page_title="Video Content Analyzer", layout="wide")
st.title("🎥 Video Content Analysis - Repetitiveness Detection")

# Sidebar navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", [
    "Upload & Transcribe",
    "Repetition Analysis",
    "Summarizer",
    "Translate",
    "NLP Features",
    "Transcription History"
])

# Initialize session state
if "video_file" not in st.session_state:
    st.session_state.video_file = None
if "transcription_text" not in st.session_state:
    st.session_state.transcription_text = ""
if "word_timestamps" not in st.session_state:
    st.session_state.word_timestamps = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = ""
if "transcription_done" not in st.session_state:
    st.session_state.transcription_done = False
if "history" not in st.session_state:
    st.session_state.history = []
if "repetition_data" not in st.session_state:
    st.session_state.repetition_data = {}

# File uploader
uploaded_file = st.sidebar.file_uploader("📤 Upload Video", type=["mp4", "mov", "avi"])
if uploaded_file:
    if uploaded_file.name != st.session_state.uploaded_filename:
        st.session_state.video_file = uploaded_file
        st.session_state.transcription_text = ""
        st.session_state.word_timestamps = None
        st.session_state.uploaded_filename = uploaded_file.name
        st.session_state.transcription_done = False
        st.session_state.repetition_data = {}

video_file = st.session_state.video_file

# Audio extraction using ffmpeg
def extract_audio_ffmpeg(video_path, audio_path):
    try:
        ffmpeg.input(video_path).output(audio_path, acodec='pcm_s16le', ar='16000').run(overwrite_output=True)
        return audio_path
    except Exception as e:
        st.error(f"❌ Error extracting audio: {e}")
        return None

# Load Whisper model
@st.cache_resource(show_spinner=False)
def load_whisper_model():
    return WhisperModel("base")

model = load_whisper_model()

# Transcription with timestamps
def transcribe_with_faster_whisper(audio_path):
    try:
        segments, _ = model.transcribe(audio_path, word_timestamps=True)
        full_text = ""
        word_timestamps = []
        for segment in segments:
            full_text += segment.text + " "
            for word in segment.words:
                word_timestamps.append((word.word.lower(), round(word.start, 2)))
        return full_text.strip(), word_timestamps
    except Exception as e:
        st.error(f"❌ Faster-whisper transcription failed: {e}")
        return None, None

# Upload & Transcribe Page
if page == "Upload & Transcribe":
    st.header("🎬 Upload Video and Transcribe Audio")
    if video_file:
        st.success("✅ Video uploaded successfully.")
        st.video(video_file)

        if st.session_state.transcription_text:
            st.text_area("📝 Transcribed Text", st.session_state.transcription_text, height=200, key="transcribed_display")
            st.download_button("📄 Download Transcription", st.session_state.transcription_text, "transcription.txt", key="download_btn_1")

        transcribe_button = st.button("🔍 Transcribe Video", disabled=st.session_state.transcription_done)

        if transcribe_button and not st.session_state.transcription_done:
            with st.spinner("🔧 Extracting audio and transcribing..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
                    tmp_vid.write(video_file.read())
                    video_path = tmp_vid.name

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                    audio_path = tmp_audio.name

                extracted_audio = extract_audio_ffmpeg(video_path, audio_path)

                if extracted_audio and os.path.exists(audio_path):
                    text, word_timestamps = transcribe_with_faster_whisper(audio_path)
                    if text:
                        st.session_state.transcription_text = text
                        st.session_state.word_timestamps = word_timestamps
                        st.session_state.transcription_done = True
                        st.session_state.history.append(text)
                        st.success("✅ Transcription complete.")
                        st.text_area("📝 Transcribed Text", text, height=200, key="transcribed_after")
                        st.download_button("📄 Download Transcription", text, "transcription.txt", key="download_btn_2")
                else:
                    st.error("❌ Audio extraction failed.")

                try:
                    os.remove(video_path)
                    os.remove(audio_path)
                except Exception as e:
                    print("Cleanup error:", e)
    else:
        st.info("Please upload a video using the sidebar.")

# Repetition Analysis Page
elif page == "Repetition Analysis":
    st.header("🔁 Repetition Analysis")
    if not st.session_state.transcription_text:
        st.warning("📝 Transcribe the video first in 'Upload & Transcribe'.")
    else:
        threshold = st.slider("🎚️ Minimum frequency to consider a word as repetitive:", min_value=2, max_value=10, value=3)

        if st.button("🔍 Analyze Repetition"):
            text = st.session_state.transcription_text
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(text.lower())
            stop_words = set(stopwords.words("english"))
            filtered_tokens = [
                word for word, pos in pos_tag(tokens)
                if word not in stop_words and pos not in ['PRP', 'PRP$', 'DT', 'IN', 'TO', 'CC', 'MD']
            ]
            fdist = FreqDist(filtered_tokens)
            repetitive = {word: freq for word, freq in fdist.items() if freq >= threshold}

            if repetitive:
                highlighted = text
                for word in repetitive:
                    highlighted = re.sub(rf'\b{re.escape(word)}\b', f'<span style="color:red">{word}</span>', highlighted, flags=re.IGNORECASE)

                st.session_state.repetition_data = {
                    "highlighted": highlighted,
                    "repetitive": repetitive
                }

        # Display stored repetition data
        if st.session_state.repetition_data:
            highlighted = st.session_state.repetition_data["highlighted"]
            repetitive = st.session_state.repetition_data["repetitive"]

            st.markdown("📝 Transcribed Text (Repetitive words highlighted):")
            st.markdown(f'<p style="white-space: pre-wrap;">{highlighted}</p>', unsafe_allow_html=True)

            repetitive_df = pd.DataFrame(repetitive.items(), columns=["Word", "Frequency"])
            st.subheader("📋 Repetitive Words")
            st.dataframe(repetitive_df)

            st.subheader("📈 Repetition Frequency Plot")
            plt.figure(figsize=(10, 4))
            sns.barplot(x=repetitive_df["Word"], y=repetitive_df["Frequency"], palette="viridis")
            plt.xticks(rotation=45)
            plt.title("Repetitive Word Frequencies")
            st.pyplot(plt)
        elif not st.session_state.repetition_data:
            st.info("No analysis performed yet.")

# Summarizer Page
elif page == "Summarizer":
    st.header("📝 Summarize Transcribed Text")
    if not st.session_state.transcription_text:
        st.warning("📝 Transcribe the video first in 'Upload & Transcribe'.")
    else:
        text = st.session_state.transcription_text
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text.strip())
        tokenizer = RegexpTokenizer(r'\w+')
        words = tokenizer.tokenize(text.lower())
        stop_words = set(stopwords.words("english"))
        filtered_words = [word for word in words if word not in stop_words]
        word_freq = FreqDist(filtered_words)
        top_words = [word for word, freq in word_freq.most_common(5)]
        sentence_scores = {}
        for sentence in sentences:
            sentence_tokens = tokenizer.tokenize(sentence.lower())
            score = sum(word_freq[word] for word in sentence_tokens if word in top_words)
            if score > 0:
                sentence_scores[sentence] = score
        summarized_sentences = heapq.nlargest(5, sentence_scores, key=sentence_scores.get)
        st.subheader("🔹 Summarized Text (Bullet Points)")
        for sentence in summarized_sentences:
            st.markdown(f"- {sentence}")

# Translate Page
elif page == "Translate":
    st.header("🌍 Translate Transcribed Text")
    if not st.session_state.transcription_text:
        st.warning("📝 Please transcribe the video first in 'Upload & Transcribe'.")
    else:
        translator = Translator()
        languages = {
            "Hindi": "hi","French": "fr", "Spanish": "es", "German": "de",
             "Chinese (Simplified)": "zh-cn",
            "Arabic": "ar", "Russian": "ru", "Japanese": "ja", "Korean": "ko"
        }
        selected_lang = st.selectbox("🌐 Choose target language", list(languages.keys()))
        lang_code = languages[selected_lang]
        if st.button("🌍 Translate"):
            with st.spinner("Translating..."):
                try:
                    translated = translator.translate(st.session_state.transcription_text, dest=lang_code)
                    st.success(f"✅ Translated to {selected_lang}:")
                    st.text_area("📘 Translated Text", translated.text, height=200)
                except Exception as e:
                    st.error(f"❌ Translation failed: {e}")

# NLP Features Page
elif page == "NLP Features":
    st.header("🧠 Additional NLP Features")
    st.write("(To be implemented: sentiment analysis, named entity recognition, keyword extraction)")
    if st.session_state.transcription_text:
        doc = nlp(st.session_state.transcription_text)
        pos_tags = [(token.text, token.pos_) for token in doc]
        pos_freq = Counter(tag for _, tag in pos_tags)
        st.subheader("📊 Part-of-Speech Tag Distribution")
        pos_df = pd.DataFrame(pos_freq.items(), columns=["POS", "Count"])
        st.bar_chart(pos_df.set_index("POS"))
    else:
        st.info("Please transcribe a video to view NLP features.")

# Transcription History Page
elif page == "Transcription History":
    st.header("📜 Transcription History")
    if not st.session_state.history:
        st.info("No transcriptions saved yet.")
    else:
        for i, past_text in enumerate(reversed(st.session_state.history), start=1):
            with st.expander(f"📄 Transcription {len(st.session_state.history) - i + 1}"):
                st.text_area("📝 Text", past_text, height=150, key=f"history_{i}")
