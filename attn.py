"""
    References:  https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
                 https://github.com/keras-team/keras/blob/master/examples
"""

from __future__ import print_function

import os
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D, Dropout, SpatialDropout1D, CuDNNLSTM, regularizers
from keras.layers import Activation, TimeDistributed, LSTM, Bidirectional
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant

from attention_with_context import AttentionWithContext

BASE_DIR = 'data'
GLOVE_DIR = os.path.join(BASE_DIR)
TEXT_DATA_DIR = os.path.join(BASE_DIR)
MAX_SEQUENCE_LENGTH = 250
MAX_NUM_WORDS = 30000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.0909090909090909090909
MAX_SENTENCE_LENGTH = 10
# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding='UTF-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

texts = []  # list of text samples
labels = []  # list of labels

files = ['train.csv', 'valid.csv']
for file_name in files:
    file = pd.read_csv(os.path.join(TEXT_DATA_DIR, file_name))
    for line in file['text']:
        texts.append(line)
    for label in file['stars']:
        labels.append(label)

print('Found %s texts.' % len(texts))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

REG_PARAM = 1e-13
l2_reg = regularizers.l2(REG_PARAM)
# word_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32')
# word_sequences = embedding_layer(word_input)
word_lstm = Bidirectional(CuDNNLSTM(150, return_sequences=True, kernel_regularizer=l2_reg))(embedded_sequences)
word_dense = TimeDistributed(Dense(200, kernel_regularizer=l2_reg))(word_lstm)
word_att = AttentionWithContext()(word_dense)
wordEncoder = Model(sequence_input, word_att)

sent_input = Input(shape=(MAX_SENTENCE_LENGTH, MAX_SEQUENCE_LENGTH), dtype='float32')
sent_encoder = TimeDistributed(wordEncoder)(sent_input)
sent_lstm = Bidirectional(CuDNNLSTM(150, return_sequences=True, kernel_regularizer=l2_reg))(sent_encoder)
sent_dense = TimeDistributed(Dense(200, kernel_regularizer=l2_reg))(sent_lstm)
sent_att = Dropout(0.5)(AttentionWithContext()(sent_dense))
preds = Dense(6, activation='softmax')(sent_att)
model = Model(sent_input, preds)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
checkpoint = ModelCheckpoint('best_han_model.h5', verbose=0, monitor='val_loss', save_best_only=True, mode='auto')
history = model.fit(x_train, y_train, verbose=2, validation_data=(x_val, y_val),
                    epochs=50, batch_size=512, callbacks=[checkpoint])
