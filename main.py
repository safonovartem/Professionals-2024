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

# Преобразование в строки в нижний регистр
low_words = no_numbers_words.lower()
#print(low_words)

#Токенизация
word_tokenize = word_tokenize(low_words)
#print(word_tokenize)

# Лемматизация
stemmer = SnowballStemmer("russian")
tokens = word_tokenize
lemmatized_words = [stemmer.stem(word) for word in tokens]
#print(lemmatized_words)

# Удаление стоп слов
stop_words = set(stopwords.words('russian'))
filtered_tokens = [word for word in lemmatized_words if word not in stop_words]
print(filtered_tokens)

# Удаления слова error
# filtered_tokens_new = re.sub(['error'], '', str(filtered_tokens))
# print(filtered_tokens_new)




