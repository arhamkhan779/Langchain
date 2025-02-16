from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.llms import Ollama


load_dotenv()
os.environ['GROQ_API_KEY']=os.getenv("groq_api")
os.environ['LANGCHAIN_API_KEY']=os.getenv("langchain_api")
os.environ['LANGCHAIN_TRACING_V2']="true"

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the user queries"),
        ("user","Question:{Question}")
    ]
)

st.title("Langchain with OLLAMA DeepSeekr1")
input_text=st.text_input("Enter the query")

llm=Ollama(model="deepseek-r1:1.5b")
output_parser=StrOutputParser()

chain=prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"Question":input_text}))