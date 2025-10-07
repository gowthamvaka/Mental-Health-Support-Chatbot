import streamlit as st
from transformers import pipeline
from typing import Tuple
import google.generativeai as genai

# === Configuration ===
GEMINI_API_KEY = "AIzaSyCgYZZumOKzlW06ouhbKURTDpObTtyfNyc"

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-pro')

st.set_page_config(page_title="AI Mental Health Companion (Gemini)", page_icon=":brain:")

@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_analyzer = load_sentiment_pipeline()

def analyze_sentiment(text: str) -> Tuple[str, float]:
    result = sentiment_analyzer(text)[0]
    return result["label"], result["score"]

def generate_chat_response(user_message: str, sentiment: str) -> str:
    if sentiment == "NEGATIVE":
        prompt = (
            f"A student is feeling anxious and says: \"{user_message}\". "
            "Respond empathetically and give a motivational tip."
        )
    elif sentiment == "POSITIVE":
        prompt = (
            f"A student shares an experience: \"{user_message}\". "
            "Respond with appropriate answer bby guessing his mood."
            "Give one supportive answer. Do not list multiple versions or options."
        )
    else:
        prompt = f"A student says: \"{user_message}\". Respond politely as a mental health companion."

    try:
        response = gemini_model.generate_content(prompt)
        # Sometimes Gemini returns an object, sometimes just text.
        if hasattr(response, "text"):
            return response.text.strip()
        return str(response).strip()
    except Exception as e:
        return f"Error communicating with Gemini API: {e}"

st.title("AI Mental Health Companion Chatbot (Gemini)")
st.write(
    "A friendly AI chatbot to support students' mental well-being. "
    "Share your thoughts or worries below, and receive empathetic responses & tips."
)

user_input = st.text_area(
    "How are you feeling today? (Share anything on your mind)",
    max_chars=400,
    height=100
)

if st.button("Send"):
    if user_input.strip():
        sentiment, score = analyze_sentiment(user_input)
        response = generate_chat_response(user_input, sentiment)
        st.markdown(f"**AI Companion:** {response}")
        st.markdown(f"_(Sentiment detected: {sentiment} | Confidence: {score:.2f})_")
    else:
        st.info("Please enter a message.")

st.markdown("---")
st.caption("This chatbot provides support but does not replace professional mental health advice.")
