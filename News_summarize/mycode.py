import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import pyttsx3
import threading

# --- Load environment variables ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop()

# --- LLM setup ---
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    google_api_key=api_key
)

# --- Thread-safe text-to-speech engine ---
engine = pyttsx3.init()
engine_lock = threading.Lock()

def speak_text(text):
    with engine_lock:
        engine.say(text)
        engine.runAndWait()

# --- Page config and background ---
st.set_page_config(page_title="📰 AI News Summarizer", layout="centered")

def set_bg_image(image_url):
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url('{image_url}');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
    """, unsafe_allow_html=True)

# Background image
set_bg_image("https://wallpapers.com/images/hd/news-background-goib5th14ifn0x77.jpg")

# --- Sidebar options ---
with st.sidebar:
    st.title("🛠️ Controls")
    input_mode = st.radio("Input Type", ["URL", "Raw Text"])
    st.markdown("---")
    st.markdown("✨ Made by Suprabha 💻")

# --- App title & input ---
st.title("🗞️ AI News Summarizer")
st.markdown("Enter a **news article URL** or paste **raw news text**, and get a smart AI summary.")

input_text = ""
if input_mode == "URL":
    input_url = st.text_input("Enter News Article URL")
    if input_url:
        try:
            res = requests.get(input_url, timeout=10)
            soup = BeautifulSoup(res.content, 'html.parser')
            paragraphs = soup.find_all('p')
            if not paragraphs:
                st.warning("No paragraphs found. Might be JavaScript-based content.")
            article = " ".join([para.get_text() for para in paragraphs])
            input_text = article
        except Exception as e:
            st.error(f"Failed to fetch or parse the URL. Error: {e}")
else:
    input_text = st.text_area("Paste your news article here", height=250)

# --- Summarization ---
if input_text:
    if st.button("🧠 Generate Summary"):
        with st.spinner("Summarizing the article..."):
            try:
                response = llm.invoke([HumanMessage(content=f"Summarize this news article:\n\n{input_text}")])
                summary = response.content.strip()

                if not summary:
                    st.warning("Summary is empty. Try again.")
                else:
                    st.session_state["summary"] = summary

                    st.subheader("🧾 AI-Generated Summary")
                    st.write(summary)

                    st.download_button("📥 Download Summary", summary, file_name="summary.txt")
            except Exception as e:
                st.error(f"Error generating summary: {e}")
else:
    st.info("Please enter a news URL or paste raw text above.")

# --- Floating Read Aloud Button ---
if "summary" in st.session_state and st.session_state["summary"]:
    st.markdown("""
        <style>
        .read-btn {
            position: fixed;
            bottom: 30px;
            right: 30px;
            background-color: #ff4b4b;
            color: white;
            padding: 15px 20px;
            border-radius: 50px;
            font-size: 20px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
            z-index: 100;
            cursor: pointer;
            transition: 0.3s ease;
        }
        .read-btn:hover {
            background-color: #e53935;
        }
        </style>
        <a href="#" class="read-btn" id="readSummary">🔊</a>
        <script>
            const readBtn = window.parent.document.getElementById("readSummary");
            if (readBtn) {
                readBtn.addEventListener("click", function() {
                    window.parent.postMessage({ type: 'read_summary' }, "*");
                });
            }
        </script>
    """, unsafe_allow_html=True)

    # Manual trigger (fallback)
    if st.button("🔊 Read Aloud"):
        threading.Thread(target=speak_text, args=(st.session_state["summary"],), daemon=True).start()
