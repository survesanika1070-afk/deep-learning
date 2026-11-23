import os
import streamlit as st
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GOOGLE_API_KEY")

# Connect to Gemini
client = genai.Client(api_key=api_key)

# Streamlit page
st.set_page_config(
    page_title="GenAI Chatbot",
    page_icon="🤖"
)

st.title("🤖 My Generative AI Chatbot")
st.write("Ask me anything!")

# User input
question = st.text_input("Enter your question:")

# Generate answer
if st.button("Ask"):
    if question:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=question
        )

        st.write("### 🤖 AI Response")
        st.write(response.text)
    else:
        st.warning("Please enter a question.")