import streamlit as st
import requests
import logging
import os
from duckduckgo_search import DDGS  # Using DuckDuckGo search library
from langchain.embeddings import HuggingFaceEmbeddings
from langgraph.graph import START, END, StateGraph
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Refresh messages if switching to Edu-Researcher
if st.session_state.get("active_function") != "Edu-Researcher":
    st.session_state.messages = []
    st.session_state.active_function = "Edu-Researcher"

# Sidebar configuration
with st.sidebar:
    st.header("Researcher Configuration")
    st.markdown("[Get HuggingFace Token](https://huggingface.co/settings/tokens)")
    st.info("Using DuckDuckGo search for web results")
    system_message = st.text_area(
        "System Message",
        value="You are an assistant for research. Use the retrieved web snippets to answer the query concisely.",
        height=100
    )
    max_tokens = st.slider("Max Tokens", 10, 4000, 300)
    temperature = st.slider("Temperature", 0.1, 4.0, 0.3)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.6)

# Set up embedding model (for context retrieval)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("Edu-Researcher")
st.caption("Powered by DuckDuckGo Search, LangGraph, and Hugging Face Inference API")

# Input research query
query_input = st.text_input("Enter your research query:")

# Define our research state as a dictionary
# The state will hold: query, sources, snippets, and response
ResearchState = Dict[str, Any]

def search_web_node(state: ResearchState) -> ResearchState:
    # Use DuckDuckGo to search for the query (max 3 results) using DDGS
    sources = []
    snippets = []
    with DDGS() as ddgs:
        results = ddgs.text(state["query"], max_results=3)
        for res in results:
            sources.append(res.get("href", ""))
            snippet = res.get("body") or res.get("title", "")
            snippets.append(snippet)
    state["sources"] = sources
    state["snippets"] = snippets
    return state

def generate_answer_node(state: ResearchState) -> ResearchState:
    # Combine retrieved snippets into a context string
    context = "\n\n".join(state.get("snippets", []))
    full_prompt = (
        f"{system_message}\n\n"
        f"Context: {context}\n\n"
        f"Query: {state['query']}\n\n"
        f"Please provide a succinct and complete answer within {max_tokens} tokens."
    )
    # Query the Hugging Face API using the selected model endpoint
    model_endpoint = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-72B-Instruct"
    headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
    logger.info(f"Sending request to {model_endpoint} with prompt: {full_prompt}")
    response = requests.post(model_endpoint, headers=headers, json={
        "inputs": full_prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "return_full_text": False
        }
    })
    logger.info(f"Received response: {response.status_code}, {response.text}")
    try:
        output = response.json()
    except requests.exceptions.JSONDecodeError:
        logger.error(f"Failed to decode JSON response: {response.text}")
        output = None
    if output and isinstance(output, list) and len(output) > 0 and "generated_text" in output[0]:
        state["response"] = output[0]["generated_text"].strip()
    else:
        state["response"] = "No response generated - please try again."
    return state

def render_message(content):
    # Render as LaTeX if enclosed by $$, else as markdown.
    if content.strip().startswith("$$") and content.strip().endswith("$$"):
        st.latex(content.strip()[2:-2])
    else:
        st.markdown(content)

# Build the state graph using langgraph
builder = StateGraph(dict)
builder.add_node("search_web", search_web_node)
builder.add_node("generate_answer", generate_answer_node)
builder.add_edge(START, "search_web")
builder.add_edge("search_web", "generate_answer")
builder.add_edge("generate_answer", END)
graph = builder.compile()

if query_input:
    # Initialize state with the query
    initial_state = {"query": query_input}
    with st.spinner("Searching the web and generating response..."):
        result_state = graph.invoke(initial_state)
    if result_state:
        render_message(result_state["response"])
        st.subheader("Sources:")
        for src in result_state.get("sources", []):
            st.write(src)
    else:
        st.error("No response generated.")
    st.session_state.messages.append({"role": "researcher", "content": result_state.get("response", "No answer.")})