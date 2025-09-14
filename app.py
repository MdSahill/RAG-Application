import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

# ===============================
# LOAD ENV
# ===============================
load_dotenv(override=True) 
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("❌ OPENAI_API_KEY not found in .env file")
    st.stop()

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="Data Science Q&A Bot", page_icon="🤖", layout="wide")

# ===============================
# LOAD PDF + VECTORSTORE
# ===============================
@st.cache_resource
def load_vectorstore():
    loader = PyPDFLoader("Data Science.pdf")
    docs = loader.load()

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.create_documents([doc.page_content for doc in docs])

    return FAISS.from_documents(chunks, embeddings)

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# ===============================
# LLM + PROMPT
# ===============================
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, streaming=True)

prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you don't know.

    {context}
    Question: {question}
    """,
    input_variables=["context", "question"],
)

# ===============================
# SIDEBAR FOR CHAT HISTORY
# ===============================
with st.sidebar:
    st.title("💬 Chat History")

    # Initialize chat storage
    if "chats" not in st.session_state:
        st.session_state.chats = {"Default Chat": []}
        st.session_state.current_chat = "Default Chat"

    # New chat button
    if st.button("➕ New Chat"):
        new_name = f"Chat {len(st.session_state.chats) + 1}"
        st.session_state.chats[new_name] = []
        st.session_state.current_chat = new_name

    # Select chat
    chat_names = list(st.session_state.chats.keys())
    selected_chat = st.radio("Select Chat", chat_names, index=chat_names.index(st.session_state.current_chat))
    st.session_state.current_chat = selected_chat

# ===============================
# MAIN CHAT UI
# ===============================
st.title("🤖 Data Science Q&A Chatbot")
st.write("Ask questions about the Data Science/AI/ML.")

# Get current chat
messages = st.session_state.chats[st.session_state.current_chat]

# Display chat history
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if user_question := st.chat_input("Ask a question..."):
    messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Retrieve docs
    retrieved_docs = retriever.invoke(user_question)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    final_prompt = prompt.invoke({"context": context_text, "question": user_question})

    # Stream assistant response
    with st.chat_message("assistant"):
        response_text = ""
        response_stream = llm.stream(final_prompt)
        placeholder = st.empty()
        for chunk in response_stream:
            response_text += chunk.content  # ✅ Fix: use .content
            placeholder.markdown(response_text + "▌")
        placeholder.markdown(response_text)

    messages.append({"role": "assistant", "content": response_text})
