GROQ_API_KEY="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"
from langchain_groq import ChatGroq
import json
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document
import faiss
import time
import pickle

# Initialize the LLM
# llm = Ollama(
#     model="llama3",
#     temperature=0  
# )
llm = ChatGroq(
        groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192", temperature=0.2
    )

# Load JSON data
def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Convert JSON data to documents (flatten and join text fields)
def json_to_documents(json_data):
    documents = []
    i=0
    for entry in json_data:
        text = (
            f"Calendar Year/Month: {entry['Calendar Year/Month']}, "
            f"Distribution Channel: {entry['Distribution Channel']}, "
            f"Sales document: {entry['Sales document']}, "
            f"Customer: {entry['Customer']}, "
            f"Name: {entry['Name']}, "
            f"Material: {entry['Material']}, "
            f"Description: {entry['Description']}, "
            f"Pricesegment: {entry['Pricesegment']}, "
            f"Product category: {entry['Product category']}, "
            f"Product sub-category: {entry['Product sub-category']}, "
            f"Sales: {entry['Sales']}, "
            f"Volumes: {entry['Volumes']}"
        )
        documents.append(Document(page_content=text))
        i = i+1
    return documents

# Load JSON data
json_data = load_json_data("transactions.json")
documents = json_to_documents(json_data)


######################## creating chunks ########################
def create_chunks(documents):
    text_splitter = CharacterTextSplitter()
    text_chunks = text_splitter.split_documents(documents)
    print(text_chunks)
    return text_chunks

text_chunks = create_chunks(documents)
######################## Writing embeddings ########################
def create_embeddings(documents):
    print("embedding is started")
    start_time = time.time()
    embeddings = HuggingFaceEmbeddings()
    knowledge_base = FAISS.from_documents(documents, embeddings)
    faiss.write_index(knowledge_base.index, 'faiss_index.bin')
    print("embedding is created")

    with open('faiss_components.pkl', 'wb') as f:
        pickle.dump({
            'embedding_function': embeddings.embed_query,
            'docstore': knowledge_base.docstore,
            'index_to_docstore_id': knowledge_base.index_to_docstore_id
        }, f)

    print(f"FAISS index and components saved successfully.")
    end_time = time.time()
    print(f"Time to get response for the query: {end_time - start_time:.2f} seconds")

create_embeddings(documents)
################## Retrieve embeddings #############################
def retrieve_embeddings():
    index = faiss.read_index('faiss_index.bin')
    with open('faiss_components.pkl', 'rb') as f:
        components = pickle.load(f)
    knowledge_base = FAISS(
        index=index,
        embedding_function=components['embedding_function'],
        docstore=components['docstore'],
        index_to_docstore_id=components['index_to_docstore_id']
    )
    print("Embedding reading completed")
    return knowledge_base

knowledge_base = retrieve_embeddings()
print("started LLM")
start_time = time.time()
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=knowledge_base.as_retriever()
)
question = "how many records are there for bath faucets"
response = qa_chain.invoke({"query": question})

print(response["result"])
end_time = time.time()
print(f"Time to get response for the query: {end_time - start_time:.2f} seconds")
