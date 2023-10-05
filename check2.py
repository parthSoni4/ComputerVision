import requests

# Construct the POST request
url = "https://powerful-plum-chick.cyclic.app/chat"
headers = {"Content-Type": "application/json"}
payload = {"message": "Hello"}

# Make the POST request
response = requests.post(url, headers=headers, json=payload)

# Parse the JSON response
response_json = response.json()

# Display the chatbot's response
print(response_json["output"])
