import requests

# Define the API endpoint URL
url = "http://localhost:5000/chat"

# Prepare the request data as a dictionary
data = {
    "message": "hello"
}

# Send a POST request to the server with JSON data
response = requests.post(url, json=data)
print(response)

# Print the response from the server
print(response.json())
