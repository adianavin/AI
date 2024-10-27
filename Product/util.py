from pymongo import MongoClient

client=MongoClient("mongodb://localhost:27017/")
db=client["hansgrohe"]
collection=db["products_main"]  #products_main

def get_unique_keys():
    # Set to hold unique keys
    unique_keys = set()

# Iterate through all documents in the collection
    for document in collection.find():
    # Add keys from each document to the set
        unique_keys.update(document.keys())

# Convert set to list (optional) and print unique keys
    unique_keys_list = list(unique_keys)
    print("Unique keys in the collection:", unique_keys_list)

get_unique_keys()