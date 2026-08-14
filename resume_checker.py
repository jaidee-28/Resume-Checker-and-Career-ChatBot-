import os
import tempfile
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate

# ───────────────────────────────────────────────
#  Page Config
# ───────────────────────────────────────────────
st.set_page_config(page_title="Resume Checker (OpenRouter)", layout="wide")

# ───────────────────────────────────────────────
#  API Key Setup
# ───────────────────────────────────────────────
HARDCODED_OPENROUTER_KEY = "API kye"

# Load key safely from env, secrets, or hardcoded fallback
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    try:
        OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY")
    except Exception:
        OPENROUTER_API_KEY = None

if not OPENROUTER_API_KEY:
    OPENROUTER_API_KEY = HARDCODED_OPENROUTER_KEY

# ───────────────────────────────────────────────
#  LLM Initialization (OpenRouter)
# ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Initializing AI model...")
def get_llm():
    return ChatOpenAI(
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        model_name="deepseek/deepseek-v4-flash-0731",  # Replace with any OpenRouter model string
        temperature=0.1,
        max_tokens=2000,
    )

llm = get_llm()

# ───────────────────────────────────────────────
#  Prompt Template
# ───────────────────────────────────────────────
EVAL_PROMPT = """
You are an advanced resume evaluation assistant. Analyze the provided resume text and 
score it out of 100 based on the following criteria: clarity, relevance, format, comprehensiveness, and keywords/ATS-friendliness.

Your response MUST follow this exact structure:

1. **Score**: X/100  
2. **Strengths**:  
   • point one  
   • point two  
   • point three (minimum 3)  
3. **Weaknesses / Areas for Improvement**:  
   • point one  
   • point two  
   • point three (minimum 3)  
4. **Skills Explicitly Mentioned**:  
   • skill 1  
   • skill 2  
   ...  
5. **Recommended Additional Skills**: (to make the resume stronger / more ATS-friendly / future-proof)  
   • suggestion 1  
   • suggestion 2  
   ...  
6. **Suggested Next Career Steps / Roles**:  
   • realistic next role 1  
   • realistic next role 2  
   • longer-term direction (optional)

Be specific, honest, constructive and professional.  
Resume text:  
{context}
"""

prompt_template = PromptTemplate(
    input_variables=["context"],
    template=EVAL_PROMPT
)

# ───────────────────────────────────────────────
#  UI Layout
# ───────────────────────────────────────────────
st.title("📄 Resume Checker powered by OpenRouter")
st.markdown("Upload your resume (PDF) → get a detailed score & improvement suggestions")

col1, col2 = st.columns([3, 2])

with col1:
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF only)",
        type=["pdf"],
        accept_multiple_files=False,
        help="Only PDF files are supported at the moment"
    )

    evaluate_button = st.button("Evaluate Resume", type="primary", disabled=not uploaded_file)

if evaluate_button and uploaded_file:
    with st.spinner("Reading PDF... → Extracting text... → Asking AI to evaluate..."):
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            # Load PDF
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            context = "\n\n".join(doc.page_content for doc in documents)

            # Clean up temporary file
            os.unlink(tmp_path)

            if not context.strip():
                st.error("No readable text was extracted from the PDF.")
                st.stop()

            # Prepare chain and invoke
            chain = prompt_template | llm
            response = chain.invoke({"context": context})

            # ── Output ────────────────────────────────────────
            st.subheader("Evaluation Result")
            st.markdown(response.content)

        except Exception as e:
            st.error("An error occurred during processing.")
            st.exception(e)

# Footer / credits
st.markdown("---")
st.caption("Built with Streamlit + LangChain + OpenRouter")