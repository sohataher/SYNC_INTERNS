# Simple E-Commerce Chatbot

This is a simple AI chatbot built using Python and PyTorch. It is designed for **e-commerce websites**, such as online stores that sell coffee, snacks, or other retail products. The chatbot can handle basic customer queries through intent classification and generate helpful responses in real time.

---

## How It Works

### 1. **Intent Classification**

The chatbot uses a dataset of predefined intents stored in a JSON file. Each intent includes:

- A `tag` representing the intent (e.g., greeting, order status)
- A list of `patterns` representing possible user inputs
- A list of corresponding `responses` for the bot to choose from

### 2. **Preprocessing**

User input is processed with the following steps:

- **Tokenization:** Splitting input sentences into individual words
- **Stemming:** Converting words to their base form
- **Bag of Words (BoW):** Representing the processed sentence as a numerical vector

### 3. **Model Training**

The model is a feedforward neural network built with PyTorch, consisting of:

- An input layer (based on the BoW vector size)
- Two hidden layers with ReLU activation
- An output layer corresponding to the number of intent classes

The model is trained using:

- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** Adam  
- **Epochs:** 200  
- **Device:** Automatically uses GPU if available, otherwise CPU


### 4. **Response Generation**

During chat, the user input is:

- Preprocessed
- Passed through the trained model
- Classified into an intent
- Matched with a random response from the relevant intent group