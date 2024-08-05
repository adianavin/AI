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
import pickle


# llm = Ollama(
#     model="llama3",
#     temperature=0  # Increased temperature to encourage more diverse responses
# )

# llm = Ollama(
#     model="gemma2:2b",
#     temperature=0  # Increased temperature to encourage more diverse responses
# )

llm = ChatGroq(
        groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192", temperature=0
)


# Convert CSV data to documents
csvlength = 0
# Load CSV data
loader = CSVLoader(
    file_path="2021 P01 Jan.csv",
    csv_args={
        "delimiter": ",",
        "quotechar": '"',
        "fieldnames": ["Calendar Year/Month","Distribution Channel","Sales document","Customer","Name","Material","Description","Pricesegment","Product category","Product sub-category","Sales","Volumes"],
    },
)

# loader = CSVLoader(
#     file_path="test.csv",
#     csv_args={
#         "delimiter": ",",
#         "quotechar": '"',
#         "fieldnames": ["slno","name","city","year"],
#     },
# )


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

#create_embeddings(text_chunks)

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


# # # Verify the retrieval mechanism
def verify_retrieval(knowledge_base):
    retriever = knowledge_base.as_retriever(search_kwargs={"k": 180})
    query = "can you provide the how many Product sub-category are there?"
    results = retriever.invoke(query)
    print(f"Number of documents retrieved: {len(results)}")
    #print(results)
    # for doc in results:  # print all results for verification        
    #     print(doc.page_content)
    #     break
    
#verify_retrieval(knowledge_base)  # Adjust num_docs to match your dataset size

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
############################
print('LLM Started')
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=knowledge_base.as_retriever(search_kwargs={"k": 50}),
    )
question = "how many transactions belongs to product sub cateogory 'Bath Faucets'"
response = qa_chain.invoke({"query": question})
print(response)
print("Response from the model:", response["result"])
############################
# end_time = time.time()
# print(f"Time to get response for the query: {end_time - start_time:.2f} seconds")

# #Additional logging to help troubleshoot
# print("Retrieved documents:")
# for doc in response["documents"]:
#     print(doc.page_content)
