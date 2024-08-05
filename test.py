# import ollama

# stream = ollama.chat(
#     model='llama3',
#     messages=[{'role': 'user', 'content': 'what is sky'}],
#     stream=True,
# )

# for chunk in stream:
#   print(chunk['message']['content'], end='', flush=True)

import ollama
import pandas as pd

# Step 1: Read the Excel file using pandas
excel_file = 'D:\\Projects\\Hansgrohe\\data\\Transactional data\\2024\\2024 P06 June.xlsx'
df = pd.read_excel(excel_file)

# Step 2: Prepare data for question answering (example: concatenating text)
# Example: Concatenate text from multiple columns into one string
columns_to_concat = ['Calendar Year/Month', 'Distribution Channel', 'Sales document',
                     'Customer Name', 'Material', 'Description', 'Pricesegment',
                     'Product category', 'Product sub-category']

# Create a list of strings from selected columns
text_data_list = []

for col in columns_to_concat:
    if col in df.columns:
        text_data_list.extend(df[col].astype(str))  # Convert to string in case of non-string data

data_for_qa = ' '.join(text_data_list)

stream = ollama.chat(
    model='phi',
    messages=[{'role': 'user', 'content': data_for_qa}],
    stream=True,
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)