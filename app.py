import pickle
from flask import Flask, request, jsonify
import numpy as np
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer=PorterStemmer()
import torch
import json
import random
import torch.nn as nn

app = Flask(__name__)



class NeuralNet(nn.Module):
  # NeuralNet is a custom class
  # this will inherit a class called nn.Module
  # nn.Module is a base class for all Pytorch neural network
  def __init__(self, input_size, hidden_size, num_classes):
    # the above function is a constructor which take three parameters
    # input size - no of input features
    # hidden size is no of neurons for hidden layers
    # output layer is no of output classses
    super(NeuralNet, self).__init__()
    # network architecture
    self.l1=nn.Linear(input_size, hidden_size)
    self.l2=nn.Linear(hidden_size, hidden_size)
    self.l3=nn.Linear(hidden_size, num_classes)
     # nn.Linear - fullly connected linear layers. they specify the input and output of each layer
    self.relu=nn.ReLU()
    # rectified linear unit activation function applied after each linear layer


model = torch.load('chatbot_model.pt', map_location=torch.device('cpu'))



# Set the model in evaluation mode
model.eval()


all_words=[]
tags=[]
xy=[]



def tokenize(sentence):
  return nltk.word_tokenize(sentence)

def stem(word):
  return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
  tokenized_sentence=[stem(w) for w in tokenized_sentence]
  # this line will stem the words to their root form
  bag=np.zeros(len(all_words), dtype=np.float32)
  # in the above line we are creating a vector which is filled by zero
  # the length of vector is equal to number of words in all_words
  for idx , w in enumerate(all_words):
    if w in tokenized_sentence:
      bag[idx]=1.0
  # if the current word is in the tokenized sentence , then the corressponding feature of bag of words is 1.0


  return bag

with open('christchatbot.json','r') as f:
  intents=json.load(f)
all_words=[]
tags=[]
xy=[]
for intent in intents['intents']:
  tag=intent['tag']
  # here we are trying to extent the tag or label of that intent
  tags.append(tag)
  for pattern in intent['patterns']:
    # we are performing tokenize operation on all sentence of pattern
    # the result is stored in w variable
    w=tokenize(pattern)

    all_words.extend(w)
    xy.append((w,tag))
    # here we are storing words with their corresponding intent tags

ignore_words=['?','!','.',',']
all_words=[stem(w) for w in all_words if w not in ignore_words]
all_words=sorted(set(all_words))
# in the above code we are sorting the variable based on alphabetical order and removing duplicates

tags=sorted(set(tags))
print(tags)
print(all_words)
print(xy)


@app.route("/")
def hello():
    return "Hello"

# API endpoint for chatbot
@app.route('/chat', methods=['POST', 'GET'])
def chat():
    try:
        sentence=request.get_json()
        # print(sentence)
        sentence=sentence['message']
        # print(sentence)
        # print("here")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sentence = tokenize(sentence)
        # print(sentence)
        X = bag_of_words(sentence, all_words)
        # print(X)
        X = X.reshape(1, X.shape[0])
        # print(X)
        X = torch.from_numpy(X).to(device)
        output = model(X)
        # print("output",output)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]
        # print("tag is ", tag)

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    print({"output" : random.choice(intent['responses'])})
                    return {"output" : random.choice(intent['responses'])}
                    # return random.choice(intent['responses'])
        else:
            return "I do not understand..."


    except Exception as e:
        return jsonify({'error': str(e)})

def generate_response(user_input):
    # Preprocess user input if necessary
    # For example, tokenize and convert text to model input format
    # processed_input = preprocess(user_input)

    # Pass the processed input through the loaded model and get the response
    # For example:
    # output = model(processed_input)
    # response = postprocess(output)

    # For this example, let's assume the model directly echoes the input (a very basic example)
    response = user_input

    return response

if __name__ == '__main__':
    app.run(debug=True)
