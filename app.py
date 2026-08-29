import streamlit as st
import pdfplumber
import pytesseract
from PIL import Image
import datetime

# --- MODERNIZED LANGCHAIN IMPORTS ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 1. HOSTING UI & SETUP
# ==========================================
st.set_page_config(page_title="CAT Prep AI Pipeline", page_icon="🚀", layout="wide")
st.title("🚀 Open-Source CAT Exam Prediction & Learning Pipeline")
st.markdown("Ingest past papers, map semantic embeddings, and predict answers using free open-source LLMs.")

st.sidebar.header("Pipeline Configuration")
hf_token = st.sidebar.text_input("Hugging Face API Token (Free)", type="password")
if not hf_token:
    st.sidebar.warning("Please enter your free Hugging Face token to enable the Prediction Engine.")

# ==========================================
# 2. INGESTION & SCANNING ENGINE
# ==========================================
def extract_data_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            else:
                img = page.to_image(resolution=300).original
                text += pytesseract.image_to_string(img) + "\n"
    return text

# ==========================================
# 3. LEARNING ENGINE (Vector DB)
# ==========================================
@st.cache_resource(show_spinner=False)
def build_learning_engine(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_text(text)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

# ==========================================
# 4. PREDICTING ENGINE (Modern Architecture)
# ==========================================
def setup_predicting_engine(vector_store, token):
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=token,
        temperature=0.2,
        max_new_tokens=512
    )
    
    # Modern Prompting System
    system_prompt = (
        "You are an expert CAT exam instructor. Use the provided context to answer the user's question accurately.\n\n"
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Modern Chain Construction
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(vector_store.as_retriever(search_kwargs={"k": 4}), question_answer_chain)
    
    return qa_chain

# ==========================================
# 5. EXECUTION & DOWNLOADING ENGINE
# ==========================================
uploaded_file = st.file_uploader("Upload CAT Exam PDF (Question Paper / Syllabus)", type="pdf")

if uploaded_file is not None and hf_token:
    with st.spinner("Ingesting and Scanning PDF..."):
        raw_text = extract_data_from_pdf(uploaded_file)
        st.success("Ingestion Complete!")
        
    with st.spinner("Learning Document Patterns (Building Vector DB)..."):
        vector_store = build_learning_engine(raw_text)
        qa_chain = setup_predicting_engine(vector_store, hf_token)
        st.success("Learning Engine Ready!")
        
    st.markdown("---")
    query = st.text_input("Ask the Predicting Engine a question (e.g., 'Solve the quantitative aptitude question regarding train speeds from the document'):")
    
    if st.button("Predict / Solve"):
        with st.spinner("Predicting..."):
            # Modern Invocation Format
            response = qa_chain.invoke({"input": query})
            result_text = response['answer']
            
            st.markdown("### Prediction Output:")
            st.write(result_text)
            
            file_name = f"CAT_Prediction_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt"
            st.download_button(
                label="Download Prediction Output (TXT)",
                data=f"Query: {query}\n\nPrediction:\n{result_text}",
                file_name=file_name,
                mime="text/plain"
            )
