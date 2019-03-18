from datetime import datetime
from nltk.corpus import stopwords
from pandas import read_csv
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import lightgbm as lgb

import numpy as np
import string
import pandas as pd
import nltk
import keras

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english') + list(string.punctuation))


# -------------- Helper Functions --------------
def tokenize(text):
    """
    :param text: a doc with multiple sentences, type: str
    return a word list, type: list
    https://textminingonline.com/dive-into-nltk-part-ii-sentence-tokenize-and-word-tokenize
    e.g.
    Input: 'It is a nice day. I am happy.'
    Output: ['it', 'is', 'a', 'nice', 'day', 'i', 'am', 'happy']
    """
    tokens = []
    for word in nltk.word_tokenize(text):
        word = word.lower()
        if word not in stop_words and not word.isnumeric():
            tokens.append(word)
    return tokens


def get_sequence(data, seq_length, vocab_dict):
    """
    :param data: a list of words, type: list
    :param seq_length: the length of sequences,, type: int
    :param vocab_dict: a dict from words to indices, type: dict
    return a dense sequence matrix whose elements are indices of words,
    """
    data_matrix = np.zeros((len(data), seq_length), dtype=int)
    for i, doc in enumerate(data):
        for j, word in enumerate(doc):
            # YOUR CODE HERE
            if j == seq_length:
                break
            word_idx = vocab_dict.get(word, 1)  # 1 means the unknown word
            data_matrix[i, j] = word_idx
    return data_matrix


def read_data(file_name, input_length, vocab=None):
    """
    https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
    """
    df = pd.read_csv(file_name)
    df['words'] = df['text'].apply(tokenize)

    if vocab is None:
        vocab = set()
        for i in range(len(df)):
            for word in df.iloc[i]['words']:
                vocab.add(word)
    vocab_dict = dict()
    vocab_dict['<pad>'] = 0  # 0 means the padding signal
    vocab_dict['<unk>'] = 1  # 1 means the unknown word
    vocab_size = 2
    for v in vocab:
        vocab_dict[v] = vocab_size
        vocab_size += 1

    data_matrix = get_sequence(df['words'], input_length, vocab_dict)
    stars = df['stars'].apply(int) - 1
    return df['review_id'], stars, data_matrix, vocab


def print_time_diff(prev_timestamp):
    new_timestamp = datetime.now().timestamp()
    print("Time Taken: %.3f seconds" % (new_timestamp - prev_timestamp))
    print()


# ----------------- End of Helper Functions-----------------


def load_data(input_length):
    # Load training data and vocab
    train_id_list, train_data_label, train_data_matrix, vocab = read_data("data/train.csv", input_length)
    K = max(train_data_label) + 1  # labels begin with 0

    # Load valid data
    valid_id_list, valid_data_label, valid_data_matrix, vocab = read_data("data/valid.csv", input_length, vocab=vocab)

    # Load testing data
    test_id_list, _, test_data_matrix, _ = read_data("data/test.csv", input_length, vocab=vocab)

    print("Vocabulary Size:", len(vocab))  # 114544
    print("Training Set Size:", len(train_id_list))  # 100000
    print("Validation Set Size:", len(valid_id_list))  # 10000
    print("Test Set Size:", len(test_id_list))  # 10000
    print("Training Set Shape:", train_data_matrix.shape)  # (100000, 300)
    print("Validation Set Shape:", valid_data_matrix.shape)  # (10000, 300)
    print("Testing Set Shape:", test_data_matrix.shape)  # (10000, 300)

    # Converts a class vector to binary class matrix.
    # https://keras.io/utils/#to_categorical
    train_data_label = keras.utils.to_categorical(train_data_label, num_classes=K)
    valid_data_label = keras.utils.to_categorical(valid_data_label, num_classes=K)
    return train_id_list, train_data_matrix, train_data_label, \
           valid_id_list, valid_data_matrix, valid_data_label, \
           test_id_list, test_data_matrix, None, vocab


if __name__ == '__main__':
    # Hyperparameters
    input_length = 300
    embedding_size = 100
    hidden_size = 100
    batch_size = 100
    dropout_rate = 0.35
    learning_rate = 0.01
    total_epoch = 3

    # Load Data
    current_timestamp = datetime.now().timestamp()
    train_id_list, train_data_matrix, train_data_label, \
    valid_id_list, valid_data_matrix, valid_data_label, \
    test_id_list, test_data_matrix, _, vocab = load_data(input_length)
    print("Data Read Finished")
    print_time_diff(current_timestamp)

    # Data shape
    N = train_data_matrix.shape[0]  # 100000
    K = train_data_label.shape[1]  # 5
    input_size = len(vocab) + 2  # 114544 + 2
    output_size = K  # 5

    start_time = datetime.now().timestamp()

    df = read_csv("data/train.csv")
    text = df['text'].tolist()
    label = df['stars'].tolist()

    test_df = read_csv("data/valid.csv")
    test_ids = test_df['review_id'].tolist()
    test_text = test_df['text'].tolist()
    test_label = test_df['stars'].tolist()
    # , cool, , funny, , , text, useful,
    df = pd.concat([df, test_df])
    total_text = text + test_text
    total_label = label + test_label

    df = df.drop(['business_id', 'date', 'review_id', 'user_id', 'stars'], axis=1)
    print(df.columns)
    df['text'] = TfidfVectorizer(ngram_range=(1, 2), smooth_idf=False).fit_transform(total_text)
    print("Vector Shape: " + str(df['text'].shape))
    x_train, x_test, y_train, y_test = train_test_split(df, total_label, test_size=0.09090909090909090909,
                                                        shuffle=False)

    classifier = LinearSVC(tol=1e-1)
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)

    print("Accuracy: %f" % accuracy_score(test_label, predictions))
    print_time_diff(start_time)
