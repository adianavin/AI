import chromadb

# Connect to Chroma DB
client = chromadb.PersistentClient(path ='D:\\Projects\\AI\\ChromaDB\\')
collection_name = "test"
collection = client.get_collection(name=collection_name)   

print(collection.peek()) # returns a list of the first 10 items in the collection
print(collection.count()) # returns the number of items in the collection
print(collection)

query_result = collection.query(
    query_texts=["get me records with year 1989"], # Chroma will embed this for you,
    where={"year": "1989"},
)
# query_result = collection.query(
#     query_texts={"metadata": {"text": {"year": "1989"}}}
# )
print(query_result)