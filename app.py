import streamlit as st
import pdfplumber
import pytesseract
import google.generativeai as genai
import datetime
import json
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="CAT Mega-Context Engine", page_icon="🧠", layout="wide")
st.title("🧠 CAT Exam Knowledge Graph & Predictive Engine")
st.markdown("Upload up to 10 years of past papers. This engine maps historical trends and renders interactive data visualizations of this year's predicted exam.")

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
        full_corpus += f"\n--- PAPER: {file.name} ---\n"
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_corpus += text + "\n"
                else:
                    img = page.to_image(resolution=300).original
                    full_corpus += pytesseract.image_to_string(img) + "\n"
    return full_corpus

# --- PREDICTION & VISUALIZATION ENGINE ---
if st.button("Generate Knowledge Graph & Question Bank"):
    if not api_key or not uploaded_files:
        st.error("Please provide an API Key and at least one PDF.")
        st.stop()

    genai.configure(api_key=api_key)
    # Using 3.6 Flash for the massive 1M token context window
    model = genai.GenerativeModel('gemini-3.6-flash')

    with st.spinner("Ingesting PDFs into Mega-Context Memory..."):
        corpus = extract_mega_corpus(uploaded_files)
        
    with st.spinner("Synthesizing Knowledge Graph & Generating JSON..."):
        prompt = f"""
        You are an elite CAT Exam Data Scientist analyzing {len(uploaded_files)} past papers.
        You MUST output your entire response as a valid JSON object. Do not include markdown formatting like ```json in the output.
        
        The JSON must strictly follow this exact structure:
        {{
            "written_analysis": "Write your full text report here including the Logic Trap Analysis, the Predictive Question Bank (3 QA, 2 DILR, 2 VARC), and detailed solutions. Use standard markdown spacing like \\n\\n for readability inside this string.",
            "section_weightage": {{
                "QA": 34,
                "DILR": 32,
                "VARC": 34
            }},
            "top_topics": [
                {{"topic": "Algebra", "frequency_percentage": 25}},
                {{"topic": "Reading Comp", "frequency_percentage": 24}},
                {{"topic": "Arithmetic", "frequency_percentage": 20}},
                {{"topic": "Data Arrangements", "frequency_percentage": 15}},
                {{"topic": "Geometry", "frequency_percentage": 10}},
                {{"topic": "Number Systems", "frequency_percentage": 6}}
            ]
        }}
        
        Ensure the numbers in the JSON reflect the actual trends from the provided corpus.
        
        HISTORICAL EXAM CORPUS:
        {corpus[:800000]} 
        """
        
        # Forcing the model to output raw JSON
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        
    st.success("Pattern Analysis Complete!")
    
    # --- PARSING & PLOTTING (ROBUST RECOVERY) ---
    try:
        # Clean potential unescaped control characters or formatting bugs in the raw string
        raw_response_text = response.text.strip()
        
        # If the model accidentally wrapped it in markdown codeblocks, strip them out
        if raw_response_text.startswith("```json"):
            raw_response_text = raw_response_text[7:]
        if raw_response_text.endswith("```"):
            raw_response_text = raw_response_text[:-3]
            
        # Parse safely using strict=False to handle minor control character quirks
        data = json.loads(raw_response_text, strict=False)
        
        st.success("Pattern Analysis & Knowledge Graph Generated Successfully!")
        st.markdown("### 📊 Exam Knowledge Graph Visualizations")
        
        # Create two columns for the charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Render a Pie Chart for Section Weightage
            pie_data = data["section_weightage"]
            fig1 = px.pie(
                names=list(pie_data.keys()), 
                values=list(pie_data.values()), 
                title="Predicted Section Weightage",
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            fig1.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            # Render a Bar Chart for the Top Topics
            bar_df = pd.DataFrame(bar_data if 'bar_data' in locals() else data["top_topics"])
            fig2 = px.bar(
                bar_df, 
                x="topic", 
                y="frequency_percentage", 
                title="Highest Frequency Topics (Predicted)",
                text_auto=True,
                color="frequency_percentage",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig2, use_container_width=True)
            
        st.markdown("---")
        st.markdown("### 📝 Written Analysis & Predictive Question Bank")
        st.markdown(data["written_analysis"])
        
        # Download Button
        file_name = f"CAT_Predictive_Engine_Output_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        st.download_button(
            label="📥 Download Written Analysis (TXT)",
            data=data["written_analysis"],
            file_name=file_name,
            mime="text/plain"
        )
        
    except Exception as e:
        st.warning(f"JSON parsing encountered a minor layout hitch, but your data was salvaged! Error: {e}")
        
        # Fallback view: Display the raw text directly so you never lose an output
        st.markdown("### 📝 Fallback Raw Output View")
        st.markdown(response.text)
        
        file_name = f"CAT_Fallback_Output_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        st.download_button(
            label="📥 Download Raw Output (TXT)",
            data=response.text,
            file_name=file_name,
            mime="text/plain"
        )
        
    except json.JSONDecodeError:
        st.error("Failed to parse the AI's data structure. The corpus may have confused the JSON generation. Try again.")
        st.write(response.text) # Fallback to show raw output
