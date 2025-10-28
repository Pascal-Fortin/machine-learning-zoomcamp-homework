import requests

url = "http://127.0.0.1:8000/predict"

client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}

response = requests.post(url, json=client)

print("Status code:", response.status_code)
print("Response text:", response.text)

# Only parse JSON if response is valid
if response.status_code == 200:
    print("Predicted result:", response.json())
