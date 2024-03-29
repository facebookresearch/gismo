# Copyright (c) Meta Platforms, Inc. and affiliates 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import string
from typing import List

from hypothesis import given
from hypothesis.strategies import lists, text

from inv_cooking.datasets.vocabulary import Vocabulary


def test_vocabulary():
    vocab = Vocabulary()
    vocab.add_word("<end>")
    for word in "this is an an example".split():
        vocab.add_word(word)
    vocab.add_word("<pad>")

    assert 6 == len(vocab), "duplicates should be removed"
    assert len(vocab) == vocab.idx
    assert_symmetric(vocab)

    vocab.remove_eos()
    assert 5 == len(vocab), "<end> token should have been removed"
    assert len(vocab) == vocab.idx
    assert_symmetric(vocab)


def test_vocabulary_with_index_option():
    vocab = Vocabulary()
    vocab.add_word("<end>")
    for group in [["this"], ["is"], ["an", "the"], ["an"], ["example"]]:
        vocab.add_word_group(group)
    vocab.add_word("<pad>")

    assert len(vocab.word2idx) == 7
    assert len(vocab.idx2word) == 6

    assert 6 == len(vocab), "should be the number of different indices"
    assert len(vocab) == vocab.idx
    assert_symmetric(vocab)

    vocab.remove_eos()
    assert 5 == len(vocab), "<end> token should have been removed"
    assert len(vocab) == vocab.idx
    assert_symmetric(vocab)


@given(words=lists(text(alphabet=string.ascii_letters, min_size=1), min_size=1))
def test_always_symmetric(words: List[str]):
    vocab = Vocabulary()
    vocab.add_word("<end>")
    vocab.add_word("<pad>")
    for word in words:
        vocab.add_word(word)
    assert_symmetric(vocab)
    vocab.remove_eos()
    assert_symmetric(vocab)


def assert_symmetric(vocab: Vocabulary):
    for i in range(len(vocab)):
        words = vocab.idx2word[i]
        if not isinstance(words, list):
            words = [words]
        for word in words:
            assert i == vocab.word2idx[word], "reversible mapping expected"
