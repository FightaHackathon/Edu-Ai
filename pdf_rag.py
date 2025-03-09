import streamlit as st
import requests
import logging
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.embeddings import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state for chat history and vector store
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Refresh messages if switching to PDF Chat (pdf_rag)
if st.session_state.get("active_function") != "pdf_rag":
    st.session_state.messages = []
    st.session_state.active_function = "pdf_rag"

# Set up PDF directory and embedding model
pdfs_directory = "chat-with-pdf\pdfs"
os.makedirs(pdfs_directory, exist_ok=True)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Sidebar configuration
with st.sidebar:
    st.header("Model Configuration")
    st.markdown("[Get HuggingFace Token](https://huggingface.co/settings/tokens)")

    # Dropdown to select model
    model_options = [
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "Qwen/Qwen2.5-72B-Instruct"
    ]
    selected_model = st.selectbox("Select Model", model_options, index=0)

    system_message = st.text_area(
        "System Message",
        value="You are an assistant for question-answering tasks created by ruslanmv.com. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.Use three sentences maximum for each question and keep the answer concise.",
        height=100
    )
    max_tokens = st.slider("Max Tokens", 10, 4000, 300)
    temperature = st.slider("Temperature", 0.1, 4.0, 0.3)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.6)

# Main interface
st.title(u"\U0001F4D1 PDF Chatbot with RAG")  # fixed emoji encoding
st.caption("Powered by Hugging Face Inference API - You can configure the temperature,tokens and top-p values in the sidebar.")

# PDF upload section
uploaded_file = st.file_uploader(
    "Upload a PDF for context",
    type="pdf",
    accept_multiple_files=False
)

if uploaded_file:
    try:
        # Save uploaded PDF
        pdf_path = os.path.join(pdfs_directory, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load and process PDF
        loader = PDFPlumberLoader(pdf_path)
        documents = loader.load()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create and store vector store
        vector_store = InMemoryVectorStore.from_documents(chunks, embedding_model)
        st.session_state.vector_store = vector_store
        st.success("PDF processed and indexed successfully!")
    except PermissionError:
        st.error(f"Permission denied: Unable to save the file to {pdf_path}. Please check the directory permissions.")
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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

# Handle user input
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.spinner("Generating response..."):
            # Check if vector store is available
            if not st.session_state.vector_store:
                st.error("Please upload a PDF first to provide context.")
                st.stop()
            
            # Retrieve relevant documents
            vector_store = st.session_state.vector_store
            related_docs = vector_store.similarity_search(prompt, k=3)
            
            # Build context
            context = "\n\n".join([doc.page_content for doc in related_docs])
            
            # Prepare full prompt with an instruction for brevity and completeness
            full_prompt = (
                f"{system_message}\n\n"
                f"Context: {context}\n\n"
                f"{prompt}\n\n"
                f"Please provide a succinct, direct, and complete answer within {max_tokens} tokens, without extra reasoning steps. Use three sentences maximum for each question and keep the answer concise."
            )
            
            # Prepare API payload
            payload = {
                "inputs": full_prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "return_full_text": False
                }
            }

            # Query API
            api_url = f"https://api-inference.huggingface.co/models/{selected_model}"
            output = query(payload, api_url)

            # Handle response
            if output and isinstance(output, list) and len(output) > 0:
                if 'generated_text' in output[0]:
                    assistant_response = output[0]['generated_text'].strip()
                    
                    with st.chat_message("assistant"):
                        st.markdown(assistant_response)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_response
                    })
                else:
                    st.error("Unexpected response format from the model")
            else:
                st.error("No response generated - please try again")
                
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")