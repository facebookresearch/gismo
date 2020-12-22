import string
from typing import List

from hypothesis import given
from hypothesis.strategies import text, lists

from inv_cooking.datasets.vocabulary import Vocabulary


def test_vocabulary():
    vocab = Vocabulary()
    vocab.add_word("<end>")
    vocab.add_word("<pad>")
    for word in "this is an an example".split():
        vocab.add_word(word)

    assert 6 == len(vocab), "duplicates should be removed"
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
        assert i == vocab.word2idx[vocab.idx2word[i]], "reversible mapping expected"
