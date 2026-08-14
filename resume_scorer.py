import os
import tempfile
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

# ───────────────────────────────────────────────
#   CONFIG & API KEY SETUP
# ───────────────────────────────────────────────
HARDCODED_OPENROUTER_KEY = "API kye"

# Safely attempt to pull key from Environment or Secrets
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    try:
        OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY")
    except Exception:
        OPENROUTER_API_KEY = None

# Fallback to hardcoded key if neither source provided one
if not OPENROUTER_API_KEY:
    OPENROUTER_API_KEY = HARDCODED_OPENROUTER_KEY

if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY not found. Please set it in secrets or environment.")
    st.stop()

# Prompt Template
PROMPT_TEMPLATE = """You are an expert resume scorer and ATS optimization specialist with deep knowledge of recruitment practices across industries.

Task: Carefully analyze how well the candidate's resume matches the job description below. Base EVERY statement, score, and suggestion **strictly and exclusively** on the content actually present in the provided resume and job description. Do NOT invent, assume, or add any experience, skills, tools, achievements, or facts that are not explicitly written in the resume.

Job Description:
{job_description}

Candidate's Resume:
{context}

Produce the analysis using **exactly** the following structure and headings (do not add/remove sections, do not change headings):

Score: [integer]/100  
Overall Match: [integer]%  

Keywords matched:  
• [bullet list of important keywords/phrases from JD that DO appear in the resume]  

Missing keywords:  
• [bullet list of important/hard-required keywords/phrases from JD that are completely absent or extremely weakly represented in the resume]  

Readability Score: [integer]/100  
ATS Compatibility Score: [integer]/100  

2-liner summary:  
[One strong, concise sentence summarizing the overall fit]  
[One strong, concise sentence naming the single biggest current weakness]

Skill gap analysis:  
• [Bullet points – clear skill/tool/experience gaps, phrased as "Missing / weak: X → needed for Y part of the role"]  
• Focus on the most impactful gaps only (4–8 bullets max)

Overall improvement suggestions:  
• [Prioritized, actionable bullet points – strongest bang-for-buck improvements first]  
• Include both content (what to add/strengthen) and formatting/ATS tips

Industry specific feedback:  
• [2–5 bullets with observations tailored to this role’s industry / function – e.g. emphasis on certifications, specific metrics, project types, modern tools, regulatory knowledge, etc. Only include points that are genuinely relevant to the JD]

Scoring rubrics to follow (use your judgment applying these):
• Score (0–100)           → weighted combination of keyword presence, skill relevance, experience recency & level, achievements quantification, role progression
• Overall Match %         → rough estimated chance of passing initial ATS + recruiter screen
• Readability             → clarity, grammar, formatting, length, action verbs, density of fluff
• ATS Compatibility       → presence of standard section headings, keyword density (not stuffing), no tables/graphics, machine-readable layout cues

Be honest, direct, and constructive. If the match is very poor, say so clearly.
"""

# ───────────────────────────────────────────────
#   STREAMLIT APP UI
# ───────────────────────────────────────────────

st.set_page_config(page_title="Resume Scorer (OpenRouter)", layout="wide")

st.title("📄 Resume Matcher & Scorer")
st.markdown("Upload your resume (PDF) and paste the job description to get a detailed match analysis powered by OpenRouter.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Job Description")
    job_description = st.text_area(
        "Paste the full job description here",
        height=320,
        placeholder="Responsibilities...\nRequirements...\nSkills...\n",
        key="jd_input"
    )

with col2:
    st.subheader("Your Resume")
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"], key="resume_uploader")

    if uploaded_file is not None:
        st.success("Resume uploaded ✓")

# ── Analyze Button Execution ────────────────────────────────────────

if st.button("Analyze Resume Match", type="primary", disabled=not (uploaded_file and job_description.strip())):

    with st.spinner("Extracting resume text..."):
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            context = "\n\n".join(doc.page_content for doc in documents)

            os.unlink(tmp_path)  # Clean up temporary file

        except Exception as e:
            st.error(f"Could not read the PDF: {e}")
            st.stop()

    if not context.strip():
        st.error("No readable text found in the resume PDF.")
        st.stop()

    # Build final prompt
    prompt = PROMPT_TEMPLATE.format(
        job_description=job_description.strip(),
        context=context.strip()
    )

    with st.spinner("Analyzing with AI via OpenRouter (this can take 20–60 seconds)..."):

        try:
            # Initialize ChatOpenAI for OpenRouter
            chat = ChatOpenAI(
                openai_api_key=OPENROUTER_API_KEY,
                openai_api_base="https://openrouter.ai/api/v1",
                model_name="deepseek/deepseek-v4-flash-0731",  # Replace with any OpenRouter model string
                temperature=0.2,                                # Low randomness for consistent scoring
                max_tokens=2200
            )

            response = chat.invoke(prompt)
            analysis_text = response.content

            st.subheader("📊 Resume Analysis Result")
            st.markdown(analysis_text)

        except Exception as e:
            st.error(f"API error: {str(e)}")
            if "rate limit" in str(e).lower():
                st.warning("Rate limit reached — please wait a few minutes and try again.")
            elif "authentication" in str(e).lower() or "401" in str(e):
                st.error("Invalid or expired OpenRouter API key.")