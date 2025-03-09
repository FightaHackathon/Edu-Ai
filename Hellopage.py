import streamlit as st
import os

st.set_page_config(
    page_title="Welcome to Edu_AI",
    page_icon="👋",
)

# Sidebar navigation
st.sidebar.title("Navigation")
# Updated navigation options to include AI Researcher
page = st.sidebar.selectbox("Go to", ["Welcome", "PDF Chatbot with RAG", "Edu Scraper with RAG", "Edu Researcher"])

if page == "Welcome":
    st.write("# Welcome to Edu_AI 👋")
    st.sidebar.success("Select a feature from the sidebar to get started.")

    st.markdown(
        """
        Edu_AI is a streamlit web application that is powered by artificial intelligence
        functions for both research and educational purposes. 
        **👈 Select a demo from the sidebar** to see some examples
        of what Edu_AI can do!
        
### Made by the talented batch 10 UIT Section-A students group 7
        """
    )
    st.image("webp.png", caption="Welcome to Edu_AI")
elif page == "PDF Chatbot with RAG":
    # Run pdf_rag.py
    with open(os.path.join(os.path.dirname(__file__), 'pdf_rag.py')) as f:
        exec(f.read())
elif page == "Edu Scraper with RAG":
    # Run Edu-Scraper.py
    with open(os.path.join(os.path.dirname(__file__), 'Edu-Scraper.py')) as f:
        exec(f.read())
elif page == "Edu Researcher":
    # Run Edu-Researcher.py
    with open(os.path.join(os.path.dirname(__file__), 'Edu-Researcher.py')) as f:
        exec(f.read())