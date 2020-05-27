import numpy as np


# the final embedding must contain all (and only) the words that the tokenizer has detected, in the order in which they
# are indexed. Words missing from the embedding file should be mapped to the zero vector (?)
def load_glove_embedding(embedding_file, embedding_size, word_index):
    words = set(word_index.keys())
    n_words = len(words)
    embedding_dict = {}
    with open(embedding_file, 'r', encoding='utf8') as file:
        for line in file.readlines():
            splitLine = line.split(' ')
            word = splitLine[0]
            if word in words:  # we only load the word if it's in the tokenizer word index
                word_vector = np.asarray(splitLine[1:], dtype='float32')
                embedding_dict[word] = word_vector

    print(len(embedding_dict.items()), "out of", len(words), "words present in embedding")
    missing_words = [w for w in words if w not in embedding_dict.keys()]
    print(len(missing_words), missing_words)

    embedding_matrix = np.zeros((n_words + 1, embedding_size))
    for word, index in word_index.items():
        if embedding_dict.get(word) is not None:
            embedding_matrix[index, :] = embedding_dict[word]
    return embedding_matrix
