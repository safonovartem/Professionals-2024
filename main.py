import io
import os
import shutil
#import tensforflow as tf
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re
import string
from nltk import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from razdel import tokenize, sentenize
from pymorphy2 import MorphAnalyzer
from navec import Navec
import slovnet

nltk.download("stopwords", quiet=True) # поддерживает удаление стоп-слов
nltk.download('punkt', quiet=True) # делит текст на список предложений
nltk.download('wordnet', quiet=True) # проводит лемматизацию

# Добавляю датасет Data
data = pd.read_csv('data.csv')
parse_dates = ['Text']
#print(data)

# Добавляю датасет Data
demo = pd.read_csv('outer_msgs_demo.csv')
parse_dates = ['Text']
#print(demo)

# Преобрзование в строку data
x = re. sub (' [^a-zA-Z] ', '', str (data))
#print(x)

# Преобрзование в строку demo
y = re.sub(' [^a-zA-Z] ', '', str (demo))
#print(y)

datasets = y + x
#print(datasets)

# Удаление пунктуации
without_punct_words= datasets.translate(str.maketrans('', '', string.punctuation))
#print(without_punct_words)

# Удаление чисел
numbers = r'[0-9]'
no_numbers_words = re.sub(numbers, '', without_punct_words)
#print(no_numbers_words)
#
# # Преобразование в строки в нижний регистр
# low_words = no_numbers_words.lower()
# #print(low_words)

#Токенизация
word_tokenize = word_tokenize(no_numbers_words)
#print(word_tokenize)

# Лемматизация
stemmer = SnowballStemmer("russian")
tokens = word_tokenize
lemmatized_words = [stemmer.stem(word) for word in tokens]
#print(lemmatized_words)

# Удаление стоп слов
stop_words = set(stopwords.words('russian'))
filtered_tokens = [word for word in lemmatized_words if word not in stop_words]
# new_filtered_tokens = set(filtered_tokens)
print(filtered_tokens)


# Морфологический анализ
def segment_text(doc: str|dict) -> dict:
    if isinstance(doc, str):
        doc = {"text": doc}

    doc["tokens"] = []

    for sent in sentenize(doc["text"]):
        doc["tokens"].append([_.text for _ in tokenize(sent.text)])

    return doc

analyzer = MorphAnalyzer()
stop_words = stopwords.words("russian")
pos = {'NOUN','ADJF','ADJS','VERB','INFN','PRTF','PRTS'}

def extract_candidates(doc: dict, stop_words: list = stop_words, pos: set = pos) -> dict:
    res = set()
    for sent in doc["tokens"]:
        for token in sent:
            if token in stop_words or token in res:
                continue

            parsed = analyzer.parse(token)[0]
            if parsed.tag.POS not in pos:
                continue
            res.add(token)
    doc["candidates"] = res

    return doc


# Синтактический анализ
navec = Navec.load("navec_news_v1_1B_250K_300d_100q.tar")
syntax = slovnet.Syntax.load("slovnet_syntax_news_v1.tar")
syntax.navec(navec)

def syntax_collocations(doc: dict, sytax: slovnet.api.Syntax = syntax) -> dict:
    syntax_colloc = []
    for sent in doc["tokens"]:
        syntax_markup = syntax(sent)

        sent_word_id = {}
        for token in syntax_markup.tokens:
            sent_word_id[token.id] = token.text

        for token in syntax_markup.tokens:
            if token.head_id != '0' and token.text in doc["candidates"]:
                syntax_colloc.append(sent_word_id[token.head_id] + ' ' + token.text)
    doc["collocations"] = set(syntax_colloc)

    return doc

doc = segment_text(datasets)
doc = extract_candidates(doc)
doc = syntax_collocations(doc)
a = list(doc["collocations"])
#print(a)

New_dataset = a + filtered_tokens
print(New_dataset)