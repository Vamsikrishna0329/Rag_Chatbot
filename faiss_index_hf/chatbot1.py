import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from typing import List

# ------------------ CONFIG ------------------
os.environ["GROQ_API_KEY"] = ""  # Replace with your key
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3-8b-8192"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 250
CHROMA_DB_DIR = "chroma_db"

# ------------------ LOAD DOCUMENTS ------------------
def load_pdfs(uploaded_files):
    docs = []
    os.makedirs("temp", exist_ok=True)
    for file in uploaded_files:
        temp_path = os.path.join("temp", file.name)
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())
        loader = PyPDFLoader(temp_path)
        docs.extend(loader.load())
    return docs

# ------------------ SPLIT DOCUMENTS ------------------
def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)

# ------------------ CUSTOM RETRIEVER WITH NEXT CHUNK ------------------
def custom_retrieve_with_next_chunk(vectorstore, query: str, k: int = 4) -> List:
    """Retrieve top-k chunks and also fetch the next chunk after each match."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    results = retriever.get_relevant_documents(query)
    
    all_docs = vectorstore._collection.get(include=["metadatas", "documents"])
    doc_texts = all_docs["documents"]
    doc_metas = all_docs["metadatas"]
    
    extended_results = []
    seen_indexes = set()
    
    for doc in results:
        if doc.page_content not in doc_texts:
            continue
        idx = doc_texts.index(doc.page_content)
        # Add current chunk
        if idx not in seen_indexes:
            extended_results.append(doc)
            seen_indexes.add(idx)
        # Add next chunk if exists
        if idx + 1 < len(doc_texts) and (idx + 1) not in seen_indexes:
            from langchain.schema import Document
            next_doc = Document(page_content=doc_texts[idx + 1], metadata=doc_metas[idx + 1])
            extended_results.append(next_doc)
            seen_indexes.add(idx + 1)
    
    return extended_results

# ------------------ CREATE QA CHAIN ------------------
def create_qa_chain(vectorstore):
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0.3,
        max_tokens=1024
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

# ------------------ STREAMLIT UI ------------------
st.set_page_config(page_title="Groq RAG Chatbot (Chroma) - Multi Page Retrieval", layout="wide")
st.title("📄 STUDYMATE-AI LEARNING ASSISTANT")
st.markdown("## Learning Assistant")
st.write("Ask me anything about your studies")

# Centered illustration - use placeholder image or static asset URL
st.image(r"C:\Users\vamsi\Documents\PVamsikrishna\faiss_index_hf\image.png", width=500)

# Welcome message
st.markdown(
    """
    # Welcome to StudyMate AI
    Your intelligent learning companion powered by advanced AI. Upload your study materials, ask questions, and get personalized explanations tailored to your learning style.
    """
)


if os.path.exists(CHROMA_DB_DIR):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    qa_chain = create_qa_chain(vectorstore)
else:
    vectorstore = None
    qa_chain = None

st.sidebar.title("StudyMate AI")
st.sidebar.write("Your intelligent learning companion")
with st.sidebar:
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing documents..."):
        docs = load_pdfs(uploaded_files)
        splits = split_docs(docs)
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectorstore = Chroma.from_documents(splits, embeddings, persist_directory=CHROMA_DB_DIR)
        vectorstore.persist()
        qa_chain = create_qa_chain(vectorstore)
    st.success("Documents processed successfully!")

if qa_chain:
    query = st.text_input("Ask a question:")
    if query:
        with st.spinner("Retrieving multiple pages..."):
            related_docs = custom_retrieve_with_next_chunk(vectorstore, query, k=4)
            context_text = "\n\n".join([doc.page_content for doc in related_docs])
            prompt = f"Answer the question using the context below:\n\n{context_text}\n\nQuestion: {query}"
            llm = ChatGroq(model=LLM_MODEL, temperature=0.3, max_tokens=1024)
            answer = llm.invoke(prompt)
            st.write("### Answer")
            st.write(answer.content)
            st.write("### Sources")
            for doc in related_docs:
                st.write(f"- {doc.metadata.get('source', 'Unknown')}")
else:
    st.info("Please upload PDFs or ensure ChromaDB exists.")
