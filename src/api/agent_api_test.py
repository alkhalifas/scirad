import requests
import json

url = "http://localhost:5001/summarize"

payload = {
    "keywords": ["AI", "protein", "folding"],
    "description": "Recent advancements using artificial intelligence to accurately predict complex protein-folding structures.",
    "model_name": "gpt-3.5-turbo",
    "temperature": 0.5,
    "top_p": 1.0,
    "summary_word_count": 300,
    "prompting_method": "Chain of Thought",
    "enable_ranking": False
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, headers=headers, json=payload)

if response.status_code == 200:
    data = response.json()
    print("Summary:", data.get("summary"))
else:
    print("Error:", response.status_code, response.text)
