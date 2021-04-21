import re
import nltk
import spacy
from nltk.corpus import stopwords

#process text

en = spacy.load('en_core_web_sm')

#tokenizer

def tokenize(text):
    tokens = en.tokenizer(text)
    return [token.text.lower() for token in tokens if not token.is_space]

#remove punctuations

def remove_punctuation(words):
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words 

#remove stop words

def remove_stopwords(words):
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def normalize(words):
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    return words

def process_text(text):
    return ' '.join(normalize(tokenize(text)))