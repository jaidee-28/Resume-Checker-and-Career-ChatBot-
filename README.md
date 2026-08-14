# Resume-Checker-and-Career-ChatBot-
# 🚀 Resume Genie

**Resume Genie** is an all-in-one AI-powered toolkit designed to accelerate job applications and career development. Built with **Streamlit**, **LangChain**, and **OpenRouter**, this application offers four core tools to analyze resumes, score candidate-job matches, generate tailored cover letters, and provide interactive career coaching.

---

## ✨ Features

- **✉️ Cover Letter Generator**: Generates customized, professional cover letters (300–450 words) aligned strictly with a target job description and your resume.
- **📊 Resume-JD Matcher**: Scores your resume against a specific job description, highlighting keyword matches, missing required skills, ATS compatibility, and tailored improvement suggestions.
- **🔍 Standalone Resume Evaluator**: Performs a comprehensive audit of your resume—scoring structure, strengths, weaknesses, ATS readiness, and career next steps.
- **💬 Career Coach Chatbot**: An interactive assistant powered by system-persisted resume context to help you prepare for interviews, formulate career moves, and refine application materials.

---

## 🛠️ Tech Stack

- **Frontend / UI**: [Streamlit](https://streamlit.io/)
- **AI Framework**: [LangChain](https://www.langchain.com/) (`langchain-openai`, `langchain-community`, `langchain-core`)
- **LLM Provider**: [OpenRouter API](https://openrouter.ai/)
- **Document Processing**: `PyPDFLoader` via `pypdf`
- **Language**: Python 3.9+

---

## 📂 Project Structure

```text
.
├── main.py                # Main Streamlit application entry point
├── requirements.txt       # Python dependencies
├── logo.png               # Sidebar logo (optional)
└── README.md              # Project documentation
