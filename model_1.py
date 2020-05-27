import h5py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn import metrics
from utils import load_glove_embedding

# config
EMBEDDING_FILE = "embeddings/glove.840B.300d.txt"
TRAINING_DATA_FILE = "data/train.csv"

embedding_size = 300  # how big is each word vector
max_words = 20000  # how many unique words to use (i.e num rows in embedding vector)
max_qst_length = 100  # max number of words in a question to use

###
# data preparation
train = pd.read_csv(TRAINING_DATA_FILE)
ids = train['qid'].values
X = train['question_text'].values
y = train['target'].values

print("accuracy baseline : ", 1 - round(sum(y) / len(y), 3), "% of questions are sincere")

ids_train, ids_test, X_train, X_test, y_train, y_test = train_test_split(ids, X, y, test_size=0.2, random_state=2020)
del X, y  # save RAM

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)
# the words to index mapping used by the tokenizer
word_index = {w: idx for (w, idx) in tokenizer.word_index.items() if idx < max_words}
words = word_index.keys()
n_words = len(words)

X_train = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_qst_length)
X_test = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_qst_length)

# load embedding
# embedding_matrix = load_glove_embedding(EMBEDDING_FILE, embedding_size, word_index)
# with h5py.File("saved_variables/embedding_matrix.h5", 'w') as file:
#     file.create_dataset("embedding_matrix", data=embedding_matrix)

with h5py.File("saved_variables/embedding_matrix.h5", 'r') as file:
    embedding_matrix = file["embedding_matrix"][:]

embedding = tf.keras.layers.Embedding(n_words + 1, embedding_size,
                                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                      trainable=False)
del embedding_matrix  # save RAM

# model
model = tf.keras.Sequential()
model.add(embedding)
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True)))
model.add(tf.keras.layers.GlobalMaxPool1D())
model.add(tf.keras.layers.Dense(16, activation="relu"))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
model.summary()
model.compile(optimizer="adam",
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=["accuracy"])

model.fit(X_train, y_train, epochs=1,
          batch_size=512,
          validation_data=(X_test, y_test))

y_test_pred_proba = model.predict(X_test, batch_size=1024, verbose=1)
y_test_pred_proba = y_test_pred_proba.flatten()
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    y_test_pred = [pp > thresh for pp in y_test_pred_proba]
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(y_test, y_test_pred)))
