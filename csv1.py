GROQ_API_KEY="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"
import pandas as pd
from langchain_groq import ChatGroq
from pymongo import MongoClient

# Replace with your MongoDB connection string
client = MongoClient('mongodb://localhost:27017/')
db = client['hansgrohe']

llm = ChatGroq(
        groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192", temperature=0
    )
transactions_collection = db['transactions']
products_main_collection = db['products_main']
def get_transactions(query):
    # Example query - customize as needed
    return list(transactions_collection.find(query))

def get_products(query):
    # Example query - customize as needed
    return list(products_main_collection.find(query))

def generate_response(prompt):
    response = llm.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']

def handle_query(user_query):
    if "transaction" in user_query.lower():
        query = {}  # Customize this based on the query
        data = get_products(query)
        prompt = f"User query: {user_query}\nData: {data}\nGenerate a response based on this data."
    elif "product" in user_query.lower():
        query = {}  # Customize this based on the query
        data = get_products(query)
        prompt = f"User query: {user_query}\nData: {data}\nGenerate a response based on this data."
    else:
        prompt = f"User query: {user_query}\nGenerate a response based on this query."

    response = generate_response(prompt)
    return response

user_query = "Tell me about the product having spray patterns"
response = handle_query(user_query)
print(response)