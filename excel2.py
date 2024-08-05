import json
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document
import faiss
import time
import pickle

# Initialize the LLM
llm = Ollama(
    model="llama3",
    temperature=0  
)

# Load JSON data
def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Convert JSON data to documents (flatten and join text fields)
def json_to_documents(json_data):
    documents = []
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
    return documents

# Load JSON data
json_data = load_json_data("transactions.json")
print(f"Total records loaded: {len(json_data)}")

# Convert JSON data to documents
documents = json_to_documents(json_data)
print(f"Total documents created: {len(documents)}")
print("Sample documents:", documents[:5])

# Create chunks from documents
def create_chunks(documents):
    text_splitter = CharacterTextSplitter()
    text_chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(text_chunks)}")
    print("Sample chunks:", text_chunks[:5])
    return text_chunks

text_chunks = create_chunks(documents)

# Create embeddings from chunks
def create_embeddings(documents):
    print("Embedding creation started")
    start_time = time.time()
    embeddings = HuggingFaceEmbeddings()
    knowledge_base = FAISS.from_documents(documents, embeddings)
    
    # Verify embedding creation
    print("Number of embeddings created:", knowledge_base.index.ntotal)

    faiss.write_index(knowledge_base.index, 'faiss_index.bin')
    print("FAISS index created")

    with open('faiss_components.pkl', 'wb') as f:
        pickle.dump({
            'embedding_function': embeddings.embed_query,
            'docstore': knowledge_base.docstore,
            'index_to_docstore_id': knowledge_base.index_to_docstore_id
        }, f)

    print("FAISS index and components saved successfully.")
    end_time = time.time()
    print(f"Time to create embeddings: {end_time - start_time:.2f} seconds")

create_embeddings(text_chunks)

# Retrieve embeddings
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

# Verify the retrieval mechanism
def verify_retrieval(knowledge_base, num_docs):
    retriever = knowledge_base.as_retriever()
    query = "how many records are there"
    results = retriever.get_relevant_documents(query, k=num_docs)
    print(f"Number of documents retrieved: {len(results)}")
    for doc in results[:5]:  # print first 5 results for verification
        print(doc.page_content)
    
verify_retrieval(knowledge_base, num_docs=100)

# Query the model
print("Started LLM")
start_time = time.time()
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=knowledge_base.as_retriever()
)
question = "how many records are there"
response = qa_chain.invoke({"query": question, "num_docs": 100})

print("Response from the model:", response["result"])
end_time = time.time()
print(f"Time to get response for the query: {end_time - start_time:.2f} seconds")
