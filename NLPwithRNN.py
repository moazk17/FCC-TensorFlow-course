from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.utils import pad_sequences
import tensorflow as tf
import os
import numpy as np

VOCAB_SIZE = 88584

MAX_LEN = 250
BATCH_SIZE = 64

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=VOCAB_SIZE)

train_data = pad_sequences(train_data, MAX_LEN)
test_data = pad_sequences(test_data, MAX_LEN)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
history = model.fit(train_data, train_labels, epochs=1, validation_split=0.2)

word_index = imdb.get_word_index()
def encode_text(text):
    tokens = tf.keras.preprocessing.text.text_to_word_sequence(text)
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    return pad_sequences([tokens], MAX_LEN)[0]

text = "that movie was just amazing, so amazing"
encoded = encode_text(text)

reverse_word_index = {value: key for (key, value) in word_index.items()}
def decode_integers(integers):
    PAD = 0
    text = ""
    for num in integers:
        if num != PAD:
            text += reverse_word_index[num] + " "
    return text[:-1]

print(decode_integers(encoded))

def predict(text):
    encoded_text = encode_text(text)
    pred = np.zeros((1, 250))
    pred[0] = encoded_text
    result = model.predict(pred)
    print(result[0])

positive_review = "awesome movie! really loved it and would great watch it again because it was amazingly great"
predict(positive_review)

negative_review = "that movie really sucked. I hated it and wouldn't watch it again. Was one of the worst things I've ever watched"
predict(negative_review)