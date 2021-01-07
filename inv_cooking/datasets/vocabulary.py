# Copyright (c) Facebook, Inc. and its affiliates.
#
# Code adapted from https://github.com/facebookresearch/inversecooking
#
# This source code is licensed under the MIT license found in the
# LICENSE file in https://github.com/facebookresearch/inversecooking
from typing import Iterable


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word: str) -> int:
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
        return self.idx

    def add_word_group(self, words: Iterable[str]):
        words = [word for word in words if word not in self.word2idx]
        if words:
            for word in words:
                self.word2idx[word] = self.idx
            self.idx2word[self.idx] = words
            self.idx += 1
        return self.idx

    def remove_eos(self):
        # get word id to remove
        id_to_remove = self.word2idx["<end>"]

        # remove word and shift all subsequent ids
        length = len(self.idx2word)
        for i in range(id_to_remove, length - 1):
            word_aux = self.idx2word[i + 1]
            if isinstance(word_aux, list):
                for el in word_aux:
                    self.word2idx[el] = i
            else:
                self.word2idx[word_aux] = i
            self.idx2word[i] = word_aux

        # remove last idx
        del self.idx2word[length - 1]
        # remove eos word
        del self.word2idx["<end>"]
        self.idx -= 1

    def __call__(self, word: str):
        if not word in self.word2idx:
            return self.word2idx["<pad>"]
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)
