import pandas as pd
import ollama

# Step 1: Read the Excel file using pandas
excel_file = 'D:\\Projects\\Hansgrohe\\data\\Transactional data\\2024\\2024 P06 June.xlsx'
df = pd.read_excel(excel_file)

# Step 2: Prepare data for question answering (example: concatenating text)
columns_to_concat = ['Calendar Year/Month', 'Distribution Channel', 'Sales document',
                     'Customer Name', 'Material', 'Description', 'Pricesegment',
                     'Product category', 'Product sub-category']

text_data_list = []
for col in columns_to_concat:
    if col in df.columns:
        text_data_list.extend(df[col].astype(str).tolist())

# Break down the data into smaller chunks
data_chunks = [text_data_list[i:i+10] for i in range(0, len(text_data_list), 10)]

# Step 3: Ask questions using ollama
questions = [
    "What is the total sales for this period?",
    "Who are the top customers?",
    "What product category had the highest sales?",
    # Add more questions as needed
]

messages = []
for chunk in data_chunks:
    messages.append({'role': 'user', 'content': ' '.join(chunk)})

messages.extend([{'role': 'user', 'content': question} for question in questions])

stream = ollama.chat(
    model='phi',
    messages=messages,
    stream=True,
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)