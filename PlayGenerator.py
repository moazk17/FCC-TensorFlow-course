from keras.preprocessing import sequence
import tensorflow as tf
import os
import keras
import numpy as np

path_to_file = keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
 
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
print('Length of text: {} characters'.format(len(text)))
 
vocab = sorted(set(text))
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

def text_to_int(text):
    return np.array([char2idx[c] for c in text])

text_as_int = text_to_int(text)
print("Text:", text[:13])
print("Encoded:", text_to_int(text[:13]))

def int_to_text(ints):
    try:
        ints = ints.numpy()
    except:
        pass
    return ''.join(idx2char[ints])

seq_length = 100
xamples_per_epoch = len(text)//(seq_length+1)
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)
# for x,y in dataset.take(2):
#     print("\n\nExample\n")
#     print("input")
#     print(int_to_text(x))
#     print("\n\noutput\n")
#     print(int_to_text(y))

BATCH_SIZE = 64
VOCAB_SIZE = len(vocab) 
EMBEDDING_DIM = 256
RNN_UNITS = 1024
BUFFER_SIZE = 10000

data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = keras.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape = [batch_size, None]),
        keras.layers.LSTM(rnn_units, return_sequences = True,  stateful=True, 
        recurrent_initializer = 'glorot_uniform'),
        keras.layers.Dense(vocab_size)
    ])
    return model
model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
model.summary() 