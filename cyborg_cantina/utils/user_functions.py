import re
from collections import Counter

import spacy
from nltk import pos_tag as nltk_pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# ---- spaCy: load once, but keep it simple ----
# NOTE: en_core_web_sm has NO real word vectors; similarity will be weak.
nlp = spacy.load("en_core_web_sm")

# ---- NLTK stopwords ----
stop_words = set(stopwords.words("english"))

def preprocess(input_sentence: str):
    """Lowercase, remove punctuation, tokenize, remove stopwords."""
    input_sentence = input_sentence.lower()
    input_sentence = re.sub(r"[^\w\s]", "", input_sentence)
    tokens = word_tokenize(input_sentence)
    return [t for t in tokens if t not in stop_words]

def compare_overlap(user_message, possible_response):
    similar_words = 0
    for token in user_message:
        if token in possible_response:
            similar_words += 1
    return similar_words

def pos_tag(tokens):
    """Wrapper so `pos_tag` can be imported from this module."""
    return nltk_pos_tag(tokens)

def extract_nouns(tagged_message):
    message_nouns = []
    for token, tag in tagged_message:
        if tag.startswith("N"):
            message_nouns.append(token)
    return message_nouns

def compute_similarity(tokens, category):
    """
    tokens: an iterable of spaCy Tokens
    category: a spaCy Doc (or Token)
    """
    output_list = []
    for token in tokens:
        output_list.append([token.text, category.text, token.similarity(category)])
    return output_list
