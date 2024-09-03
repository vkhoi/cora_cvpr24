import nltk
import os
import json

from transformers import BertTokenizer


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.all_tokens = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['[UNK]']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def load_vocab(path):
    with open(path) as f:
        d = json.load(f)
    vocab = Vocabulary()
    vocab.word2idx = d['word2idx']
    vocab.idx2word = {}
    for i in d['idx2word']:
        vocab.idx2word[int(i)] = d['idx2word'][i]
    vocab.idx = d['idx']
    vocab.all_tokens = list(d['idx2word'].keys())
    return vocab


def get_vocab(vocab_path, ty='scratch'):
    if ty == 'scratch':
        vocab = load_vocab(vocab_path)
        tokenizer = nltk.tokenize.word_tokenize
    elif ty.startswith('bert'):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        vocab = tokenizer.vocab
    else:
        assert False
    return vocab, tokenizer
