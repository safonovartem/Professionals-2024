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

# # Удаление пунктуации
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
print(Bow)

# Получить имена функций (слова)
#print(vectorizer.get_feature_names_out())

# Векторизация (TF-IDF)

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the data
X_tfidf = vectorizer.fit_transform(New_dataset)

# Convert sparse matrix to dense matrix for better visualization
TF_vector = X_tfidf.toarray
print(TF_vector())

# Get feature names (words)
#print(vectorizer.get_feature_names_out())


# Кластеризация

# Загрузить данные из CSV (при условии, что «data.csv» содержит столбцы «тема» и «текст»)
# data = pd.read_csv('data.csv')

# # Предварительная обработка текста и векторизация
# vectorizer = TfidfVectorizer(stop_words='english')
# X = vectorizer.fit_transform(data['text'])

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

# Визуализация оценок силуэта
plt.plot(range_n_clusters, kmeans_scores, label='K-means')
#plt.plot(range_n_clusters, hierarchical_scores, label='Hierarchical')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different Clustering Algorithms')
plt.legend()
plt.show()

# Визуализация кластерных структур с помощью визуализатора силуэтов
silhouette_visualizer = SilhouetteVisualizer(KMeans(n_clusters=3, random_state=42))
silhouette_visualizer.fit(Bow)
silhouette_visualizer.show()


# Метод t-SNE

# определям скоость и модель обучения
#model = TSNE(learning_rate=100)

# обучаем модель
#transformed = model.fit_transform(x_vocab)

# результат
# x_axis = transformed[:, 0]
# y_axis = transformed[:, 1]

# plt.scatter(x_axis, y_axis, c=x_vocab)
# plt.show()




# #Добавляю test
# with open ('test (1).csv', newline = '') as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in reader:
#         print(row['text'], '|' , row['category'])
#
# test = pd.read_csv('test (1).csv')
# # темы текстов из которых буду брать текста
# topic = ['football', 'volleyball', 'autosport', 'hockey', 'tennis', 'esport', 'basketball', 'martial_arts', 'winter_sport', 'athletics', 'extreme', 'motosport', 'boardgames']
# number_of_text = 2 # по 2 текста на тему
#
# df_res = pd.DataFrame()
#
# for topic in tqdm(topic):
#     df_topic = test[test['category'] == topic][:number_of_text]
#     df_res = pd.concat(df_topic, pd.DataFrame([df_topic]), ignore_index=True)