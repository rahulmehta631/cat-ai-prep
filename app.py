import streamlit as st
import pdfplumber
import pytesseract
from PIL import Image
import google.generativeai as genai
import datetime
import json
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="CAT Exam Structural Architect", page_icon="🧬", layout="wide")
st.title("🧬 CAT Deep Structural Knowledge Graph & Blueprint-Exact Mock Generator")
st.markdown("Upload multiple years of past papers. The engine decomposes historical intra-sectional constraints and builds a **100% fresh Mock CAT Paper** matching official IIM question counts (VARC: 24, DILR: 20, QA: 22).")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Engine Configuration")
api_key = st.sidebar.text_input("Gemini API Key (Free)", type="password")

# --- INGESTION ENGINE ---
uploaded_files = st.file_uploader(
    "Upload Historical CAT Papers (Select multiple PDFs)", 
    type="pdf", 
    accept_multiple_files=True
)

def extract_mega_corpus(files):
    full_corpus = ""
    for file in files:
        full_corpus += f"\n=== HISTORICAL PAPER SOURCE: {file.name} ===\n"
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_corpus += text + "\n"
                else:
                    img = page.to_image(resolution=300).original
                    full_corpus += pytesseract.image_to_string(img) + "\n"
    return full_corpus

# --- ARCHITECTURAL GENERATION ENGINE ---
if st.button("Run Deep Structural Analysis & Generate Blueprint-Exact Mock"):
    if not api_key or not uploaded_files:
        st.error("Please provide your Google AI Studio API Key and upload at least one past paper PDF.")
        st.stop()

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3.6-flash')

    with st.spinner("Step 1/3: Parsing historical corpus and mapping intra-sectional matrices..."):
        corpus = extract_mega_corpus(uploaded_files)
        
    with st.spinner("Step 2/3: Enforcing official CAT structural constraints (VARC: 24, DILR: 20, QA: 22)..."):
        prompt = f"""
        You are an elite psychometrician and CAT Exam Convenor. You have analyzed the uploaded historical CAT papers.
        
        Your task is to build a **100% FRESH, Original Mock CAT Paper** that strictly respects the exact official CAT exam blueprint and intra-sectional proportions derived from both the official pattern and the uploaded corpus trends:
        
        OFFICIAL CAT SECTIONAL CONSTRAINTS:
        1. **VARC Section (24 Questions Total):** 
           - 4 RC Passages with 4 original questions each (16 Qs).
           - 8 Verbal Ability questions (Para Jumbles, Paragraph Summary, Odd-One-Out).
        2. **DILR Section (20 Questions Total):** 
           - Exactly 4 high-complexity sets with 5 questions each (20 Qs total), mapping the precise logic/arrangements/quantitative-reasoning trend observed in the corpus.
        3. **QA Section (22 Questions Total):** 
           - Exact sub-topic proportion allocation (Arithmetic, Algebra, Geometry, Modern Math/Number Systems) reflecting the exact weightage shifts found in your historical data analysis.
        
        You MUST output your response as a valid JSON object matching this exact schema (no markdown blocks around it):
        {{
            "structural_analysis": "Provide a rigorous technical breakdown of intra-sectional weightages, sub-topic hybridization ratios, and difficulty curves extracted from the corpus.",
            "section_weightage": {{
                "VARC (24 Qs)": 36,
                "DILR (20 Qs)": 30,
                "QA (22 Qs)": 34
            }},
            "top_hybrid_topics": [
                {{"topic": "Algebra-Function & Graph Hybrids", "frequency_percentage": 28}},
                {{"topic": "Complex Scheduling & Dynamic DILR Grids", "frequency_percentage": 25}},
                {{"topic": "Advanced Critical Reasoning RC", "frequency_percentage": 22}},
                {{"topic": "Number Theory & P&C Overlaps", "frequency_percentage": 15}},
                {{"topic": "Arithmetic Mixtures & Alligations", "frequency_percentage": 10}}
            ],
            "fresh_mock_paper": "Write out the complete, 100% brand-new, original Mock CAT Paper. Structure it clearly into Section VARC (listing the 4 RCs and 8 VA items), Section DILR (listing the 4 sets with 5 questions each), and Section QA (listing the 22 proportional questions). Provide comprehensive step-by-step solutions and answer keys for every single question. Use clean markdown formatting like \\n\\n for readability."
        }}
        
        HISTORICAL EXAM CORPUS:
        {corpus[:800000]} 
        """
        
        try:
            # response = model.generate_content(
            #     prompt,
            #     generation_config={"response_mime_type": "application/json"}
            # )
            # raw_text = response.text.strip()
            # if raw_text.startswith("```json"):
            #     raw_text = raw_text[7:]
            # if raw_text.endswith("```"):
            #     raw_text = raw_text[:-3]
                
            # data = json.loads(raw_text, strict=False)
            
            # st.success("Blueprint-Exact Mock CAT Paper Successfully Synthesized!")
            # We remove response_mime_type="application/json" to prevent JSON boundary/escape character crashes on massive text blocks
            response = model.generate_content(prompt)
            raw_text = response.text.strip()
            
            st.success("Blueprint-Exact Mock CAT Paper Successfully Synthesized!")
            
            st.markdown("---")
            st.markdown("### 📝 Blueprint-Exact Mock CAT Paper (VARC: 24 | DILR: 20 | QA: 22)")
            st.markdown(raw_text)
            
            # --- VISUALIZATION DASHBOARD ---
            st.markdown("### 📊 Official Blueprint Proportions & Topic Matrix")
            col1, col2 = st.columns(2)
            
            with col1:
                pie_data = data["section_weightage"]
                fig1 = px.pie(
                    names=list(pie_data.keys()), 
                    values=list(pie_data.values()), 
                    title="Official CAT Sectional Proportions",
                    color_discrete_sequence=px.colors.sequential.Sunsetdark
                )
                fig1.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig1, use_container_width=True)
                
            with col2:
                bar_df = pd.DataFrame(data["top_hybrid_topics"])
                fig2 = px.bar(
                    bar_df, 
                    x="topic", 
                    y="frequency_percentage", 
                    title="Intra-Sectional Hybrid-Topic Weightage",
                    text_auto=True,
                    color="frequency_percentage",
                    color_continuous_scale="Teal"
                )
                st.plotly_chart(fig2, use_container_width=True)
                
            st.markdown("---")
            st.markdown("### 🔬 Intra-Sectional Structural Intelligence")
            st.markdown(data["structural_analysis"])
            
            st.markdown("---")
            st.markdown("### 📝 Blueprint-Exact Mock CAT Paper (VARC: 24 | DILR: 20 | QA: 22)")
            st.markdown(data["fresh_mock_paper"])
            
            # --- DOWNLOADING ENGINE ---
            full_export = f"=== INTRA-SECTIONAL STRUCTURAL BLUEPRINT ===\n\n{data['structural_analysis']}\n\n\n=== BLUEPRINT-EXACT MOCK CAT PAPER & SOLUTIONS ===\n\n{data['fresh_mock_paper']}"
            # --- DOWNLOADING ENGINE ---
           
            file_name = f"Blueprint_Exact_Mock_CAT_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt"
            st.download_button(
                label="📥 Download Full Blueprint Mock Paper (.txt)",
                data=raw_text,
                file_name=file_name,
                mime="text/plain"
            )
            
        except Exception as e:
            st.error(f"Generation error: {e}")
        #     file_name = f"Blueprint_Exact_Mock_CAT_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        #     st.download_button(
        #         label="📥 Download Full Blueprint Mock Paper (.txt)",
        #         data=full_export,
        #         file_name=file_name,
        #         mime="text/plain"
        #     )
            
        # except Exception as e:
        #     st.warning(f"JSON formatting boundary detected, rendering safe raw output fallback: {e}")
        #     st.markdown(response.text)
        #     st.download_button(
        #         label="📥 Download Raw Output (.txt)",
        #         data=response.text,
        #         file_name=f"CAT_Blueprint_Output_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        #         mime="text/plain"
        #     )
