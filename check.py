import requests

# Define the API endpoint URL
url = "http://localhost:3000/chat"
# url = "https://symphonious-crisp-56c412.netlify.app/chat"


# Prepare the request data as a dictionary
data = {
    "message": "hello"
}

# Send a POST request to the server with JSON data
response = requests.post(url, json=data)
print(response)

# Print the response from the server
print(response.json())
