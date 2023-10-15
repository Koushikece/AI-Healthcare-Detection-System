# -*- coding: utf-8 -*-
"""
Created on Fri Sept 15 11:43 , 2023

@author: Koushik
"""
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import keras
import numpy as np
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('C:\\Users\\koush\\OneDrive\\Documents\\AI-based-Healthcare-Chatbot-and-Disease-Detection-System\\data.json').read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:

        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents in the corpus
        documents.append((w, intent['tag']))

        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print (len(documents), "documents")
# classes = intents
print (len(classes), "classes", classes)
# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)


pickle.dump(words,open('C:\\Users\\koush\\OneDrive\\Documents\\AI-based-Healthcare-Chatbot-and-Disease-Detection-System\\texts.pkl','wb'))
pickle.dump(classes,open('C:\\Users\\koush\\OneDrive\\Documents\\AI-based-Healthcare-Chatbot-and-Disease-Detection-System\\labels.pkl','wb'))

# Create our training data
training = []
# Create an empty array for our output
output_empty = [0] * len(classes)

for doc in documents:
    bag = []  # Initialize our bag of words
    pattern_words = doc[0]  # List of tokenized words for the pattern
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]  # Lemmatize words

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # Output is a '0' for each tag and '1' for the current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append(bag + output_row)  # Concatenate bag and output_row

# Shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# Split train_x (patterns) and train_y (intents)
train_x = training[:, :len(words)]
train_y = training[:, len(words):]

print("Training data created")

# Create model - 3 layers
model = keras.Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model using SGD optimizer with corrected parameter names
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('C:\\Users\\koush\\OneDrive\\Documents\\AI-based-Healthcare-Chatbot-and-Disease-Detection-System\\model.h5', hist)

print("Model created")