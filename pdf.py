GROQ_API_KEY="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"
import os;
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
# llm = Ollama(
#     model="llama3",
#     temperature=0  
# )
llm = ChatGroq(
        groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192", temperature=0.7
    )

loader = UnstructuredFileLoader("test.pdf")
documents = loader.load()
#print(documents)

#create document chunks
#chunk_size: The maximum number of characters in each chunk. If not specified, a default value is used.
#chunk_overlap: The number of characters that overlap between consecutive chunks. This can help maintain context between chunks.
#text_splitter = CharacterTextSplitter(separator="/n", chunk_size=200, chunk_overlap=50)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text_chunks = text_splitter.split_documents(documents)

#loading the vector embedding models = create text into vectors
# embeddings = HuggingFaceEmbeddings(
#     model_name="distilbert-base-uncased", # sentence-transformers/all-MiniLM-L6-v2
#     tokenizer_kwargs={"padding": True, "truncation": True},
#     embedding_kwargs={"batch_size": 16}
# )  
embeddings = HuggingFaceEmbeddings()
knowledge_base = FAISS.from_documents(text_chunks, embeddings)

#retrival QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever = knowledge_base.as_retriever()
)

question = "can you make translate this document to Hindi"

response = qa_chain.invoke({"query":question})

print(response["result"])


