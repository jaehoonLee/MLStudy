import torch
from collections import Counter
import numpy as np
import random

def preprocess(text):
    # Replace punctuation with tokens so we can use them in our model
    text = text.lower()
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    # text = text.replace('\n', ' <NEW_LINE> ')
    text = text.replace(':', ' <COLON> ')
    words = text.split()

    # Remove all words with  5 or fewer occurences
    word_counts = Counter(words)
    trimmed_words = [word for word in words if word_counts[word] > 5]

    return trimmed_words


def create_lookup_tables(words):
    """
    Create lookup tables for vocabulary
    :param words: Input list of words
    :return: Two dictionaries, vocab_to_int, int_to_vocab
    """
    word_counts = Counter(words)
    # sorting the words from most to least frequent in text occurrence
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)

    # create int_to_vocab dictionaries
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    # If I use the most common and create int_to_vocab,
    # then it can override ii since ii represent # of occurences for the word.

    return vocab_to_int, int_to_vocab


def get_target(words, idx, window_size=5):
    targets = []
    size = random.randint(1, window_size + 1)
    start = 0 if idx - size < 0 else idx - size
    end = len(words) - 1 if idx + size >= len(words) else idx + size
    for i in range(start, end + 1):
        if words[i] != idx:
            targets.append(words[i])

    return targets


def get_batches(words, batch_size, window_size=5):
    n_batches = len(words)//batch_size
    words = words[:n_batches*batch_size]# only full batch

    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx:idx+batch_size]
        for ii in range(len(batch)):
            batch_x = batch[ii]
            batch_y = get_target(batch, ii, window_size)
            y.extend(batch_y)
            x.extend([batch_x]*len(batch_y))
        yield x, y


def cosine_similarity(embedding, valid_size=16, valid_window=100, device='cpu'):
    embed_vectors = embedding.weight
    magnitudes = embed_vectors.pow(2).sum(dim=1).sqrt().unsqueeze(0)

    valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
    valid_examples = np.append(valid_examples, random.sample(range(1000, 1000+valid_window), valid_size//2))
    valid_examples = torch.LongTensor(valid_examples).to(device)

    valid_vector = embedding(valid_examples)
    similarities = torch.mm(valid_vector, embed_vectors.t())/magnitudes # 16 * 300 and 300 * 63641 and 16 * 63641

    return valid_examples, similarities


