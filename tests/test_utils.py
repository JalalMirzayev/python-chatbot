import pytest
import numpy
from utils import tokenize
from utils import stem_tokens
from utils import bag_of_words
from utils import get_words


@pytest.mark.parametrize('input_text, expected_output', [
    ('how long does eighty-seven shipping to new york cost?', ['how', 'long', 'does', 'eighty', 'seven', 'shipping', 'to', 'new', 'york', 'cost']),
    ('', [])
])
def test_tokenize(input_text, expected_output):
    assert tokenize(input_text) == expected_output


@pytest.mark.parametrize('words, input_list, expected_output', [
    (['i', 'am', 'not', 'dog'], ['you', 'are', 'not', 'a', 'dog'], numpy.array([0.0, 0.0, 1.0, 1.0])),
    (['you', 'are', 'cat'], ['you', 'are', 'not', 'a', 'dog'], numpy.array([1.0, 1.0, 0.0])),
    (['i', 'will', 'boat'], ['you', 'are', 'not', 'a', 'dog'], numpy.array([0.0, 0.0, 0.0])),
    (['i', 'will', 'fly'], ['i', 'will', 'fly', 'i'], numpy.array([1.0, 1.0, 1.0])),
])
def test_bag_of_words(words, input_list, expected_output):
    assert numpy.array_equal(bag_of_words(words, input_list), expected_output)


@pytest.mark.parametrize('tokens, expected_output', [
    (['i', 'am', 'organizing', 'a', 'regulatory', 'party'], ['i', 'am', 'organ', 'a', 'regulatori', 'parti']),
    (['donald', 'ordered', 'a', 'bookstore'], ['donald', 'order', 'a', 'bookstor'])
])
def test_stem_tokens(tokens, expected_output):
    assert stem_tokens(tokens) == expected_output

@pytest.mark.parametrize('tokens, expected_output', [
    ([(['hi', 'i', 'am'], 'greeting'), (['hi', 'how', 'much', 'shipping'], 'shipping')], (['am', 'hi', 'how', 'i', 'much', 'shipping'], ['greeting', 'shipping'])),
    ([(['hi', 'i', 'am'], 'shipping'), (['hi', 'i', 'am'], 'shipping')], (['am', 'hi', 'i'], ['shipping'])),
    ([(['hi', 'i', 'am'], 'greeting'), (['good', 'morning'], 'greeting')], (['am', 'good', 'hi', 'i', 'morning'], ['greeting']))
])
def test_get_words(tokens, expected_output):
    assert get_words(tokens) == expected_output
