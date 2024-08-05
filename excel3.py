GROQ_API_KEY="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"
import csv
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.document_loaders import CSVLoader
import time
import faiss
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Initialize the LLM
# llm = Ollama(
#     model="llama3",
#     temperature=0.1  # Increased temperature to encourage more diverse responses
# )

llm = ChatGroq(
        groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192", temperature=0.7
)


# Convert CSV data to documents
csvlength = 0
# Load CSV data
# loader = CSVLoader(
#     file_path="2021 P01 Jan.csv",
#     csv_args={
#         "delimiter": ",",
#         "quotechar": '"',
#         "fieldnames": ["Calendar Year/Month","Distribution Channel","Sales document","Customer","Name","Material","Description","Pricesegment","Product category","Product sub-category","Sales","Volumes"],
#     },
# )

loader = CSVLoader(
    file_path="test.csv",
    csv_args={
        "delimiter": ",",
        "quotechar": '"',
        "fieldnames": ["slno","name","city","year"],
    },
)


data = loader.load()
csvlength = len(data)
print(csvlength)

#print(data)

# Create chunks from documents
def create_chunks(documents):
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    text_chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(text_chunks)}")
    #csvlength = len(text_chunks)
    return text_chunks

#text_chunks = create_chunks(data)

# Create embeddings from chunks and keep FAISS index in memory



def create_embeddings_chroma(documents):
    print("Embedding creation started")
    start_time = time.time()   
    texts = [doc.content if hasattr(doc, 'content') else str(doc) for doc in documents]
    embeddings_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
        # Create embeddings
    #embeddings = [embeddings_model.embed([doc])[0] for doc in documents]
    embeddings = embeddings_model.embed_documents(texts)
    
    # Connect to Milvus
    client = chromadb.PersistentClient(path ='D:\\Projects\\AI\\ChromaDB\\')
    collection_name = "test"
    collection = client.get_collection(name=collection_name)

    # Insert documents and their embeddings
    for i, (doc, embedding) in enumerate(zip(texts, embeddings)):
        collection.add(
            ids=[f"doc_{i}"],        # IDs should be provided as a list
            embeddings=[embedding],  # Embeddings should be provided as a list
            metadatas=[{"text": doc}]  # Metadata should also be provided as a list
        )

    # Verify the number of inserted entities
    collection_info = client.get_collection(name=collection_name)
    print(collection_info)

    # Verify embedding creation
       
    end_time = time.time()
    print(f"Time to create embeddings: {end_time - start_time:.2f} seconds")

def retrieve_from_chroma():
    # Connect to Chroma DB
    client = chromadb.PersistentClient(path ='D:\\Projects\\AI\\ChromaDB\\')
    
    # Retrieve or create collection
    collection_name = "test"
    collection = client.get_collection(name=collection_name)
    
    # Query the collection (example query to get all documents)
    query_result = collection.query()
    
    # Access embeddings from the query result
    embeddings = query_result['embeddings']
    
    return embeddings


#knowledge_base = create_embeddings(text_chunks)


# # # Verify the retrieval mechanism
# def verify_retrieval(knowledge_base):
#     retriever = knowledge_base.as_retriever(search_kwargs={"k": csvlength})
#     query = "tell me the total records in csv file"
#     results = retriever.invoke(query)
#     print(f"Number of documents retrieved: {len(results)}")
#     # for doc in results:  # print all results for verification
#     #     print(doc.page_content)
    
# verify_retrieval(knowledge_base)  # Adjust num_docs to match your dataset size

# # Uncomment the following lines if you want to query the model
# # Query the model
# print("Started LLM")
# start_time = time.time()




# qa_chain = RetrievalQA.from_chain_type(
#     llm,
#     retriever=knowledge_base.as_retriever()
# )

# prompt_template = """Use the following pieces of context to answer the question at the end. 
# {context}
# Question: {question}
# """
# PROMPT = PromptTemplate(
#     template=prompt_template, input_variables=["context", "question"]
# )
# chain_type_kwargs = {"prompt": PROMPT}

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=knowledge_base.as_retriever(search_kwargs={"k": 500}),
    )
question = "how many total records are there?"
response = qa_chain.invoke({"query": question})

print("Response from the model:", response["result"])
# end_time = time.time()
# print(f"Time to get response for the query: {end_time - start_time:.2f} seconds")

# #Additional logging to help troubleshoot
# print("Retrieved documents:")
# for doc in response["documents"]:
#     print(doc.page_content)
