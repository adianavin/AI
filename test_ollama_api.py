import requests
import json

# Define the URL
url = 'http://127.0.0.1:11434/api/generate'

json_file_path = 'transactions.json'

# Read the JSON file
with open(json_file_path, 'r', encoding='utf-8') as file:
    json_data = json.load(file)

# Define the payload with the model and prompt
payload = {
    'model': 'gemma2:2B',
    'prompt': 'total number of records',
    'options': {
        'seed': 123
    },
    'data': json_data
}

# Convert the payload to JSON format
json_payload = json.dumps(payload)

# Send the POST request with headers
headers = {
    'Content-Type': 'application/json'
}

response = requests.post(url, data=json_payload, headers=headers)

# Print the response from the server
print('Response Status Code:', response.status_code)
print('Response Text:', response.text)

# Check the content type of the response
content_type = response.headers.get('Content-Type', '')
print('Response Content-Type:', content_type)