import tempfile
import os
import base64
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# --- Page Configuration ---
st.set_page_config(page_title="AI Career Coach", page_icon="💼", layout="wide")

st.title("💼 AI Career Coach & Resume Mentor")

# --- Sidebar: Configuration & API Key ---
with st.sidebar:
    st.header("⚙️ Configuration")
    # API key is pulled from input (or environment variable) so it's not hardcoded in code
    api_key = st.text_input(
        "OpenRouter API Key", 
        value=os.getenv("OPENROUTER_API_KEY", ""), 
        type="password",
        help="Paste your OpenRouter API Key here."
    )
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# --- Helper Function: Extract Text from PDF ---
@st.cache_data(show_spinner="Extracting PDF text...")
def extract_pdf_content(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name
    
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    os.remove(tmp_path)
    
    return "\n\n".join(doc.page_content for doc in documents)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Main Layout: Side-by-Side Setup ---
left_col, right_col = st.columns([1, 1], gap="medium")

# ==================== LEFT COLUMN: PDF UPLOADER & PREVIEW ====================
with left_col:
    st.subheader("📄 Upload & View Resume")
    uploaded_file = st.file_uploader("Upload your resume (PDF format)", type=["pdf"])

    if uploaded_file:
        bytes_data = uploaded_file.getvalue()
        
        # Display embedded PDF inside an iframe preview
        base64_pdf = base64.b64encode(bytes_data).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="650" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    else:
        st.info("👆 Please upload a PDF resume above to start chatting with your coach.")

# ==================== RIGHT COLUMN: AI CHATBOT INTERFACE ====================
with right_col:
    st.subheader("💬 Career Coach Chat")

    if not uploaded_file:
        st.warning("Awaiting resume upload...")
    elif not api_key:
        st.error("⚠️ Please enter your OpenRouter API Key in the left sidebar to activate the AI Coach.")
    else:
        # Extract context from PDF
        context = extract_pdf_content(uploaded_file.getvalue())

        # Define System Message
        system_message = SystemMessage(
            content=f"""
            You are a professional career coach and resume mentor.

            You help with:
            - Career Guidance
            - Resume Improvements
            - Interview Preparation
            - Job Search Strategy
            - Skill Gap Analysis

            Candidate Resume:
            {context}
            """
        )

        # Initialize LLM with user-provided API key safely
        llm = ChatOpenAI(
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            model_name="deepseek/deepseek-v4-flash-0731",
            streaming=True
        )

        # Render Previous Messages
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(message.content)

        # Handle New User Input
        if user_input := st.chat_input("Ask your career coach anything..."):
            # Display User Message
            with st.chat_message("user"):
                st.markdown(user_input)
            
            st.session_state.chat_history.append(HumanMessage(content=user_input))

            # Build Full Message Stream
            full_messages = [system_message] + st.session_state.chat_history

            # Display Streaming Assistant Response
            with st.chat_message("assistant"):
                response_container = st.empty()
                full_response = ""

                try:
                    for chunk in llm.stream(full_messages):
                        full_response += chunk.content
                        response_container.markdown(full_response + "▌")
                    response_container.markdown(full_response)
                except Exception as e:
                    st.error(f"Error communicating with AI: {e}")

            st.session_state.chat_history.append(AIMessage(content=full_response))
