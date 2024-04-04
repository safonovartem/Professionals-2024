import io
import csv
import os
import shutil
from typing import Union
from sklearn.feature_extraction.text import TfidfVectorizer
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
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from numpy import asarray
from sklearn.preprocessing import OrdinalEncoder
from sklearn import datasets
from  sklearn.manifold import TSNE
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

nltk.download("stopwords", quiet=True) # поддерживает удаление стоп-слов
nltk.download('punkt', quiet=True) # делит текст на список предложений
nltk.download('wordnet', quiet=True) # проводит лемматизацию
pd.set_option('display.expand_frame_repr', False)


#Добавляю датасет Data
data = pd.read_csv('data.csv')
parse_dates = ['Text']
#print(data)

#Добавляю датасет Data
demo = pd.read_csv('outer_msgs_demo.csv')
parse_dates = ['Text']
demo = demo['Text']
#print(demo)


#Преобрзование в строку data
x = re. sub (' [^a-zA-Z] ', '', str (data))
#print(x)

# Преобрзование в строку demo
y = re.sub(' [^a-zA-Z] ', '', str (demo))
#print(y)
datasets = x + y
#print(datasets)


# Удаление пунктуации
without_punct_words= datasets.translate(str.maketrans('', '', string.punctuation))
#print(without_punct_words)


#Удаление чисел
numbers = r'[0-9]'
no_numbers_words = re.sub(numbers, '', without_punct_words)
#print(no_numbers_words)


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
#print(filtered_tokens)


# Морфологический анализ
def segment_text(doc: Union[str, dict]) -> dict:
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
#print(New_dataset)


# Векторизация (Bag of words)

# Инициализировать CountVectorizer
vectorizer = CountVectorizer()

# Сопоставьте и преобразуйте данные
X_bow = vectorizer.fit_transform(New_dataset)

# Преобразуйте разреженную матрицу в плотную для лучшей визуализации
Bow = X_bow.toarray()
#print(Bow)

# Получить имена функций (слова)
#print(vectorizer.get_feature_names_out())


# Векторизация (TF-IDF)
# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the data
X_tfidf = vectorizer.fit_transform(New_dataset)

# Convert sparse matrix to dense matrix for better visualization
TF_vector = X_tfidf.toarray
#print(TF_vector())

# Get feature names (words)
#print(vectorizer.get_feature_names_out())


# Кластеризация

# Определить диапазон кластеров для K-средних
range_n_clusters = [2, 3, 4, 5, 6]

# Инициализируем списки для хранения оценок силуэтов
kmeans_scores = []
hierarchical_scores = []

# Выполняем кластеризацию K-средних
for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(Bow)
    silhouette_avg = silhouette_score(Bow, cluster_labels)
    kmeans_scores.append(silhouette_avg)

# Выполняем иерархическую кластеризацию
hierarchical = AgglomerativeClustering(n_clusters=3)
hierarchical_labels = hierarchical.fit_predict(Bow)
hierarchical_score = silhouette_score(Bow, hierarchical_labels)
hierarchical_scores.append(hierarchical_score)

# Визуализация оценок силуэта k -средних
plt.plot(range_n_clusters, kmeans_scores, label='K-means')
plt.xlabel('Количество кластеров')
plt.ylabel('Оценка силуэта')
plt.title('Оценка силуэта для различных алгоритмов кластеризации')
plt.legend()
plt.show()

# Визуализация оценок силуэта иерархической кластеризации
plt.plot(hierarchical_scores, label='Hierarchy-means')
plt.xlabel('Количество кластеров')
plt.ylabel('Оценка силуэта')
plt.title('Оценка силуэта для различных алгоритмов кластеризации')
plt.legend()
plt.show()


# # Визуализация кластерных структур с помощью визуализатора силуэтов
# silhouette_visualizer = SilhouetteVisualizer(KMeans(n_clusters=3, random_state=42))
# silhouette_visualizer.fit(Bow)
# silhouette_visualizer.show()


# Загрузка теста и разделение данных
test = pd.read_csv('test (1).csv')
test_text = test['text']
test_text = test_text[0:46]
test_topic = test['category']
test_topic = test_topic[0:46]
print(test_text,test_topic)

# Разделите набор данных на обучающий и тестовый наборы (80 % обучение, 20 % тестирование)
X_train, X_test, y_train, y_test = train_test_split(test_text, test_topic, test_size=0.2, random_state=42)

# Два метода векторизации
vectorizers = {
    'CountVectorizer': CountVectorizer(),
    'TfidfVectorizer': TfidfVectorizer()
}

# Три алгоритма для машинного обучения
models = {
    'Logistic Regression': LogisticRegression(),
    'Multinomial Naive Bayes': MultinomialNB(),
    'Random Forest': RandomForestClassifier()
}

# Обучение модели
for vec_name, vectorizer in vectorizers.items():
    for model_name, model in models.items():
        # Vectorize the training and test data
        X_train_vec = vectorizer.fit_transform(X_train.values.astype('U'))
        X_test_vec = vectorizer.transform(X_test.values.astype('U'))

        # Обучение
        model.fit(X_train_vec, y_train)

        # Делайте прогнозы на тестовом наборе
        y_pred = model.predict(X_test_vec)

        # Оценка точность
        accuracy = accuracy_score(y_test, y_pred)

        # Print the results
        print(f'{vec_name} + {model_name} Точность: {accuracy:.4f}')