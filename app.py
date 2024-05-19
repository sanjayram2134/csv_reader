import streamlit as st 
from langchain_community.llms import Ollama 
import pandas as pd
from pandasai import SmartDataframe
from dotenv import load_dotenv
import os
from fpdf import FPDF
import base64

from langchain_groq.chat_models import ChatGroq
from llama_index.llms.groq import Groq

load_dotenv(override=True)

llm = ChatGroq(
    model_name="mixtral-8x7b-32768", 
    api_key=os.environ["GROQ_API_KEY"]
)

st.title("Data Analysis with PandasAI")

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# File uploader
uploader_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploader_file is not None:
    data = pd.read_csv(uploader_file)
    st.write(data.head(3))
    df = SmartDataframe(data, config={"llm": llm})
    
    prompt = st.text_area("Enter your prompt:")

    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating response..."):
                # Generate bot response
                bot_response = df.chat(prompt)
                
                # Update chat history
                st.session_state.chat_history.append(f"You: {prompt}")
                st.session_state.chat_history.append(f"Bot: {bot_response}")
                
                # Clear the input box
                st.text_area("Enter your prompt:", value='', key="user_input", disabled=True)
        else:
            st.warning("Please enter a prompt!")

# Display chat history
if st.session_state.chat_history:
    st.write("### Chat History")
    for message in st.session_state.chat_history:
        st.write(message)

    # Function to create PDF with title
    def create_pdf(chat_history, dataset_name):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Title
        pdf.set_font("Arial", style="B", size=16)
        pdf.cell(200, 10, f"CSV Analysis of {dataset_name}", ln=True, align="C")
        pdf.ln(10)
        
        # Chat history
        pdf.set_font("Arial", size=12)
        for message in chat_history:
            pdf.multi_cell(0, 10, message)
        
        return pdf

    # Generate and provide a link to download the PDF
    if st.button("Download Chat History as PDF"):
        dataset_name = os.path.splitext(uploader_file.name)[0] if uploader_file else "Unnamed Dataset"
        pdf = create_pdf(st.session_state.chat_history, dataset_name)
        pdf_output = pdf.output(dest='S').encode('latin1')
        
        b64 = base64.b64encode(pdf_output).decode('latin1')
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="chat_history.pdf">Download PDF</a>'
        st.markdown(href, unsafe_allow_html=True)
