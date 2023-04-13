import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet
from chat_dataset import ChatDataset

if __name__ == '__main__':

    with open('intents.json', 'r') as f:
        intents = json.load(f)

    all_words = []
    tags = []
    x_y = []

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            word = tokenize(pattern)
            all_words.extend(word)
            x_y.append((word, tag))

    ignore_words = [',', '.', '?', '!']
    all_words = [stem(word) for word in all_words if word not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    x_train = []
    y_train = []

    for (pattern, tag) in x_y:
        bag = bag_of_words(pattern, all_words)
        x_train.append(bag)

        label = tags.index(tag)
        y_train.append(label)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # Hyper parameters
    batch_size = 4
    input_size = len(all_words)
    hidden_size = 8
    output_size = len(tags)
    learning_rate = 0.001
    epochs = 200

    dataset = ChatDataset(x_train, y_train)

    train_loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)

            outputs = model(words.to(torch.float32))
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch) % 20 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags
    }

    FILE = "data.pth"
    torch.save(data, FILE)

    print(f'Training complete. file saved to {FILE}')
