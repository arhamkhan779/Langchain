import os
import streamlit as st
import sounddevice as sd
import numpy as np
import wave
from groq import Groq
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("groq_api")

# Set up Groq client
client = Groq(api_key=GROQ_API_KEY)

# Define LLaMA 3 model in LangChain
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

# Define LangChain Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries"),
        ("user", "Question:{Question}")
    ]
)

# Chain the prompt with LLM and output parser
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Streamlit UI
st.title("🤖 AI Chatbot with Voice Input (LLama3 and whisper-large-v3-turbo)")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input (text)
user_input = st.chat_input("Type your message here...")

# Audio recording parameters
samplerate = 44100  
duration = 5  # Adjust recording duration (in seconds)

# Voice Input Button
if st.button("🎤 Speak"):
    st.write("🎙️ Recording... Speak now!")
    
    # Record audio
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()
    
    # Save audio as a WAV file
    filename = "recorded_audio.wav"
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())

    st.success("✅ Recording completed! Processing...")

    # Transcribe with Groq Whisper
    with open(filename, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=file,
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
        )

    # Extract text from transcription
    user_input = transcription.text.strip() if transcription.text else ""

    # Display transcribed text
    if user_input:
        st.subheader("📝 Transcription:")
        st.write(user_input)

# Process input (text or voice)
if user_input:
    # Display user message in chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get AI response
    ai_response = chain.invoke({"Question": user_input})

    # Display AI response
    with st.chat_message("assistant"):
        st.markdown(ai_response)
    
    # Store AI response in chat history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})