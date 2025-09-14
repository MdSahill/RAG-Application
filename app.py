
from flask import Flask, request, jsonify, session, render_template
from flask_session import Session
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

# ===============================
# FLASK APP CONFIG
# ===============================
app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# ===============================
# LOAD ENV
# ===============================
load_dotenv(override=True)
# ===============================
# LOAD PDF + VECTORSTORE
# ===============================
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
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

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
# ROUTES
# ===============================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chats", methods=["GET"])
def get_chats():
    """Return list of available chats"""
    if "chats" not in session:
        session["chats"] = {"Default Chat": []}
        session["current_chat"] = "Default Chat"
    return jsonify({
        "chats": list(session["chats"].keys()),
        "current": session["current_chat"]
    })

@app.route("/new_chat", methods=["POST"])
def new_chat():
    """Create a new chat"""
    if "chats" not in session:
        session["chats"] = {"Default Chat": []}
    new_name = f"Chat {len(session['chats']) + 1}"
    session["chats"][new_name] = []
    session["current_chat"] = new_name
    session.modified = True
    return jsonify({"new_chat": new_name})

@app.route("/switch_chat", methods=["POST"])
def switch_chat():
    """Switch to another chat"""
    chat_name = request.json.get("chat")
    if chat_name in session.get("chats", {}):
        session["current_chat"] = chat_name
        session.modified = True
        return jsonify({"switched": chat_name})
    return jsonify({"error": "Chat not found"}), 400

@app.route("/chat", methods=["POST"])
def chat():
    """Send a message in the current chat"""
    user_question = request.json.get("question", "").strip()
    if not user_question:
        return jsonify({"error": "Question is required"}), 400

    if "chats" not in session:
        session["chats"] = {"Default Chat": []}
        session["current_chat"] = "Default Chat"

    current_chat = session["current_chat"]

    # Retrieve docs
    retrieved_docs = retriever.invoke(user_question)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    final_prompt = prompt.invoke({"context": context_text, "question": user_question})
    response = llm.invoke(final_prompt)
    response_text = response.content

    # Save chat history
    session["chats"][current_chat].append({"role": "user", "content": user_question})
    session["chats"][current_chat].append({"role": "assistant", "content": response_text})
    session.modified = True

    return jsonify({
        "answer": response_text,
        "history": session["chats"][current_chat]
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
