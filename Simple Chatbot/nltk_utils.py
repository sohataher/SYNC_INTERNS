import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):

    tokenized_sentence = [stem(word) for word in tokenized_sentence]

    bag = np.zeros(len(all_words))

    for index, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[index] = 1
    return bag
