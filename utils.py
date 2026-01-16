import json
import numpy
import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')


stemmer = nltk.stem.PorterStemmer()
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')


def tokenize(sentence: str) -> list[str]:
    return tokenizer.tokenize(sentence)


def stem_tokens(tokens: list[str]) -> list[str]:
    return [stemmer.stem(token) for token in tokens]


def bag_of_words(words: list[str], tokens: list[str]) -> list[float]:
    """
    Docstring for bag_of_words
    
    :param words: Dictionary containing all possible words.
    :param tokens: List with tokens of the sentence.
    :return: Float vector with length of words
    """

    # output = len(words) * [0.0]
    # for index, word in enumerate(words):
    #     output[index] = float(word in tokens)

    # return numpy.array(output)
    return numpy.array([float(word in tokens) for word in words])


def load_intents(path: str) -> tuple[list[str], str]:
    with open(path, 'r') as file:
        intents = json.load(file)['intents']

    output: list[tuple] = []

    for intent in intents:
        tag: str = intent['tag']
        patterns: list[str] = intent['patterns']
        for pattern in patterns:
            pattern_lower = pattern.lower()
            pattern_tokens = tokenize(pattern_lower)
            pattern_stemmed = stem_tokens(pattern_tokens)
            output.append((pattern_stemmed, tag))
    
    return output


def get_words(data) -> tuple[list[str], list[str]]:
    words = []
    tags = []
    for tokens, tag in data:
        words.extend(tokens)
        tags.append(tag)
    
    words_unique = list(set(words))
    tags_unique = list(set(tags))
    return (sorted(words_unique), sorted(tags_unique))


def prepare_data(words, data) -> list[tuple[numpy.array, str]]:
    # output: list[tuple[numpy.array, str]] = []
    # for tokens, tag in data:
    #     tokens_bag_of_words = bag_of_words(words, tokens)
    #     output.append((tokens_bag_of_words, tag))

    # return output
    return [(bag_of_words(words, tokens), tag) for tokens, tag in data]
    

if __name__ == "__main__":
    data = load_intents(path='intents.json')
    words_unique, tags_unique = get_words(data)
    prepared_data = prepare_data(words=words_unique, data=data)
    print(tags_unique)
    print(prepared_data)
