# https://github.com/TsingZ0/PFLlib/blob/master/dataset/utils/language_utils.py

import re
import numpy as np
import json
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)


def letter_to_index(letter):
    return ALL_LETTERS.find(letter)


def _one_hot(index, size):
    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec


def letter_to_vec(letter):
    index = ALL_LETTERS.find(letter)
    return _one_hot(index, NUM_LETTERS)


def word_to_indices(word):
    print('num of letters (classes):', NUM_LETTERS)
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices


def split_line(line):
    return re.findall(r"[\w']+|[.,!?;]", line)


def _word_to_index(word, indd):
    if word in indd:
        return indd[word]
    else:
        return len(indd)


def line_to_indices(line, word2id, max_words=25):
    unk_id = len(word2id)
    line_list = split_line(line)  # split phrase in words
    indl = [word2id[w] if w in word2id else unk_id for w in line_list[:max_words]]
    indl += [unk_id] * (max_words - len(indl))
    return indl


def bag_of_words(line, vocab):
    bag = [0] * len(vocab)
    words = split_line(line)
    for w in words:
        if w in vocab:
            bag[vocab[w]] += 1
    return bag


def get_word_emb_arr(path):
    with open(path, 'r') as inf:
        embs = json.load(inf)
    vocab = embs['vocab']
    word_emb_arr = np.array(embs['emba'])
    indd = {}
    for i in range(len(vocab)):
        indd[vocab[i]] = i
    vocab = {w: i for i, w in enumerate(embs['vocab'])}
    return word_emb_arr, indd, vocab


def val_to_vec(size, val):
    assert 0 <= val < size
    vec = [0 for _ in range(size)]
    vec[int(val)] = 1
    return vec


def tokenizer(text, max_len, max_tokens=32000):
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(
        map(tokenizer, iter(text)),
        specials=['<pad>', '<cls>', '<unk>', '<eos>'],
        special_first=True,
        max_tokens=max_tokens
    )
    vocab.set_default_index(vocab['<unk>'])
    text_pipeline = lambda x: vocab(tokenizer(x))

    text_list = []
    for t in text:
        tokens = [vocab['<cls>']] + text_pipeline(t) # a list of tokens
        padding = [0 for i in range(max_len - len(tokens))]  # fill the rest with 0
        tokens.extend(padding)
        text_list.append(tokens[:max_len])
    return vocab, text_list