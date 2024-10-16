GROQ_API_KEY="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"
import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import tempfile

# Streamlit UI setup
st.image("https://www.hansgrohe.com/medias/Hansgrohe-Logo-2.svg?context=bWFzdGVyfGltYWdlc3wyNjI2fGltYWdlL3N2Zyt4bWx8aW1hZ2VzL2gxZi9oNDcvOTA5NzgyMDM3MzAyMi5zdmd8YTA5ZGQ2ODk1ZmFlZGExMDIzZjI1MjI0NTJlYTJhYWRiMTdiNzRlMDIxZGRjMWUzNjE2YWU5NDExMTY1ZmEwZQ", width=350)
st.title("Hansgrohe - LLM Powered AI App")
st.write("Upload a document and ask questions to get answers based on the content.")

# File uploader for document
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Load document
    loader = UnstructuredFileLoader(temp_file_path)
    documents = loader.load()

    # Create document chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(documents)

    # Load embeddings and create knowledge base
    embeddings = HuggingFaceEmbeddings()
    knowledge_base = FAISS.from_documents(text_chunks, embeddings)

    # Initialize LLaMA model
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY, 
        model_name="llama3-8b-8192", 
        temperature=0
    )

    # Create RetrievalQA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=knowledge_base.as_retriever()
    )

    # Chat feature
    st.write("### Chat with the document")
    user_question = st.text_input("Ask a question:")
    
    if st.button("Get Answer"):
        if user_question:
            response = qa_chain.invoke({"query": user_question})
            st.write("**Answer:**", response["result"])
        else:
            st.write("Please ask a question.")

else:
    st.write("Please upload a PDF file to start.")

