# Chatbot using Pytorch
This code is an implementation of a simple chatbot for an e-commerce website using PyTorch. 
The chatbot is trained on a dataset of intents in JSON format and uses natural language processing techniques to generate responses to user input.

## Dataset
The dataset used to train the chatbot is a JSON file called intents.json. The file contains a list of intents, each of which has a tag and a list of patterns. The chatbot uses these patterns to generate responses to user input.

## Usage
The program will read in a dataset of intents from a file called intents.json and use it to train a neural network using PyTorch. 
The code first tokenizes and stems the words in the dataset, then creates a bag of words for each pattern in the dataset. The bag of words is used to train a neural network with the specified hyperparameters.
The trained model is saved to a file called data.pth. To use the chatbot, load the trained model from this file and use it to generate responses to user input.
