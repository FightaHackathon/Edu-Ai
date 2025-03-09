import streamlit as st
import requests
import logging
import os
from langchain_community.document_loaders import SeleniumURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.embeddings import HuggingFaceEmbeddings

# Disable proxy for Hugging Face endpoints
os.environ["NO_PROXY"] = "huggingface.co"

# PATCH: Force SeleniumURLLoader to use local chromedriver.exe using Service
import langchain_community.document_loaders.url_selenium as url_selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

def patched_get_driver(self):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    # Use a simple path: assume "chromedriver.exe" is in the current working directory
    service = Service("chromedriver.exe")
    return webdriver.Chrome(service=service, options=chrome_options)

url_selenium.SeleniumURLLoader._get_driver = patched_get_driver
# END PATCH

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state for chat history and vector store
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Set up embedding model (used for indexing)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Sidebar configuration (similar to pdf_rag.py)
with st.sidebar:
    st.header("Model Configuration")
    st.markdown("[Get HuggingFace Token](https://huggingface.co/settings/tokens)")
    # Updated model options list with additional option for Qwen2.5-72B-Instruct
    model_options = [
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "Qwen/Qwen2.5-72B-Instruct"
    ]
    selected_model = st.selectbox("Select Model", model_options, index=0)
    system_message = st.text_area(
        "System Message",
        value="You are an assistant for question-answering tasks created by ruslanmv.com. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum for each question and keep the answer concise.",
        height=100
    )
    max_tokens = st.slider("Max Tokens", 10, 4000, 300)
    temperature = st.slider("Temperature", 0.1, 4.0, 0.3)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.6)

# After sidebar configuration and session_state initialization
if st.session_state.get("active_function") != "Edu-Scraper":
    st.session_state.messages = []
    st.session_state.active_function = "Edu-Scraper"

# Main interface
st.title("Edu Scraper with RAG")
st.caption("Powered by Hugging Face Inference API - Configure parameters in the sidebar.")

# URL input section (instead of PDF upload)
url = st.text_input("Enter URL for scraping:")

if url:
    try:
        with st.spinner("Loading page..."):
            documents = SeleniumURLLoader(urls=[url]).load()
        with st.spinner("Splitting content..."):
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
            chunks = splitter.split_documents(documents)
        # Create and store vector store
        vector_store = InMemoryVectorStore.from_documents(chunks, embedding_model)
        st.session_state.vector_store = vector_store
        st.success("Page processed and indexed successfully!")
    except Exception as e:
        st.error(f"Error processing URL: {str(e)}")

def render_message(content):
    # If content is enclosed by $$, render as LaTeX; else as markdown.
    if content.strip().startswith("$$") and content.strip().endswith("$$"):
        st.latex(content.strip()[2:-2])
    else:
        st.markdown(content)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        render_message(message["content"])

# Function to query Hugging Face API
def query(payload, api_url):
    headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
    logger.info(f"Sending request to {api_url} with payload: {payload}")
    response = requests.post(api_url, headers=headers, json=payload)
    logger.info(f"Received response: {response.status_code}, {response.text}")
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        logger.error(f"Failed to decode JSON response: {response.text}")
        return None

# Handle user chat input
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        render_message(prompt)
    try:
        with st.spinner("Generating response..."):
            if not st.session_state.vector_store:
                st.error("Please enter a URL and process the page first.")
                st.stop()
            vs = st.session_state.vector_store
            related_docs = vs.similarity_search(prompt, k=3)
            context = "\n\n".join([doc.page_content for doc in related_docs])
            full_prompt = (
                f"{system_message}\n\n"
                f"Context: {context}\n\n"
                f"{prompt}\n\n"
                f"Please provide a succinct, direct, and complete answer within {max_tokens} tokens, without extra reasoning steps. Use three sentences maximum for each question and keep the answer concise."
            )
            payload = {
                "inputs": full_prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "return_full_text": False
                }
            }
            api_url = f"https://api-inference.huggingface.co/models/{selected_model}"
            output = query(payload, api_url)
            if output and isinstance(output, list) and len(output) > 0 and "generated_text" in output[0]:
                assistant_response = output[0]["generated_text"].strip()
                with st.chat_message("assistant"):
                    render_message(assistant_response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_response
                })
            else:
                st.error("No response generated - please try again")
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")