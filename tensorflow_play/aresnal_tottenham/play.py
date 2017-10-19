import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import string
import unicodedata
import sys


# a table structure to hold pronunciation types
punct_tbl = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))


# function to remove punctuation from a table
def remove_punctuation(text):
    return text.translate(punct_tbl)

# init the stemmer
stemmer = LancasterStemmer()

# read the json file and load the training data
with open('data.json', 'r') as json_data:
    data = json.load(json_data)

# list of all the categories to train for
categories = list(data.keys())
words = []
# a list of tuple with words in the sentence and the respective category name
docs = []

for each_category in data.keys():
    for each_sentence in data[each_category]:
        # remove punctuation from sentence
        print(each_sentence)
        # extract words from each sentence and append to the word list
        w = nltk.word_tokenize(each_sentence)
        print('tokenized words', w)
        words.extend(w)
        docs.append((w, each_category))

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words)))

print(words)
print(docs)

# created our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(categories)


for doc in docs:
    # create a bag of words for each document in the list
    bow = []
    # list of tokenized words for the pattern
    token_words = doc[0]
    # stem each word
    token_words = [stemmer.stem(word.lower()) for word in token_words]
    # create our bag of words array
    for w in words:
        bow.append(1) if w in token_words else bow.append(0)

    output_row = list(output_empty)
    output_row[categories.index(doc[1])] = 1

    # our traning set will contain a bog of words and output row with category
    training.append([bow, output_row])


random.shuffle(training)
training = np.array(training)

# trainX contains the bag of words and trainX the label/category
train_x = list(training[:,0])
train_y = list(training[:,1])

tf.reset_default_graph()

net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensobord
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

model.fit(train_x, train_y, n_epoch=2000, batch_size=8, show_metric=True)
model.save('model.tflearn')

sent_1 = "Tottenham star Danny Rose may make injury return against Real Madrid as he trains with the team at the Bernabeu ahead of Champions League clash"
sent_2 = "Tottenham Hotspur Football Club, commonly referred to simply as Tottenham"
sent_3 = "Arsenal boss Arsene Wenger could face FA charge after confronting referee following defeat at Watford"
sent_4 = "Arsenal's lack of leaders is staggering, they have no driving force and nobody to rip the opposition apart"

def get_tf_record(sentence):
    global words
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    # bag of words
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bow[i] = 1

    return (np.array(bow))

print(categories[np.argmax(model.predict([get_tf_record(sent_1)]))])
print(categories[np.argmax(model.predict([get_tf_record(sent_2)]))])
print(categories[np.argmax(model.predict([get_tf_record(sent_3)]))])
print(categories[np.argmax(model.predict([get_tf_record(sent_4)]))])