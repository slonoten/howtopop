
# coding: utf-8

import json
from math import log, exp, log1p
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import scipy
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from scipy.sparse import coo_matrix
from scipy.linalg import logm, expm
from nltk.stem.snowball import RussianStemmer
import Stemmer
from stop_words import get_stop_words
from datetime import datetime
import matplotlib

def load_post(path):
    with open(path) as json_file:
        post = json.load(json_file)
    hubs = {t['title'] : True for t in post['hubs']}   
    return [post['_id'], post['published']['$date'], post['title'], post['author']['url'], post['domain'], hubs, post['content'], post['tags']] 

def load_posts(path):
    for file_name in listdir(path):
        file_path = join(path, file_name)
        if isfile(file_path):
            yield load_post(file_path)
            
def get_image_count(html):
    return len(re.findall('<img.*?>', html))            
            
def prepare_data(path):
    data = pd.DataFrame(load_posts(path), columns = ['_id', 'published', 'title', 'author', 'domain', 'hubs', 'content', 'tags'])
    data['published'] = pd.to_datetime(data['published'])
    # Считаем и нормализуем количество изображений
    #data['image_count'] = data['content'].apply(get_image_count)
    #data['image_count'] = data['image_count'] / data['image_count'].max()
    # Считаем и нормализуем длину текста
    data['content_length'] = data['content'].str.len()
    data['sites'] = data['content'].apply(lambda html: list(re.findall('<a href="https?://(.+?)(?:/.*"|")>', html)))
    return data

test = prepare_data('./test/')
df_train = prepare_data('./train/')
target = pd.read_csv('./train_target.csv')
train = df_train.merge(target, on = '_id')
train[['_id', 'published', 'favs_lognorm']][train['domain'] == 'geektimes.ru'].to_csv('gt_favs.csv', index = False)
train[['_id', 'published', 'favs_lognorm']][train['domain'] == 'habrahabr.ru'].to_csv('habr_favs.csv', index = False)

print('делаем прогноз средней популярности статьи в тестовый период')
exec(open('favs_prediction.py').read())
habr_mean_fav = pd.read_csv('habr_favs_mean_pred.csv').fillna(0)
gt_mean_fav = pd.read_csv('gt_favs_mean_pred.csv').fillna(0)
gt_mean_fav.columns = habr_mean_fav.columns = ['date', 'favs_mean60', 'favs_mean_pred']
gt_mean_fav['date'] = pd.to_datetime(gt_mean_fav['date'])
habr_mean_fav['date'] = pd.to_datetime(habr_mean_fav['date'])
gt_mean_fav.set_index('date', inplace = True)
habr_mean_fav.set_index('date', inplace = True)

def get_mean_fav(timestamp, domain):
    return (habr_mean_fav if domain == 'habrahabr.ru' else gt_mean_fav).loc[timestamp.date(), 'favs_mean60']

train['favs_meanlog'] = train.apply(lambda row: log1p(get_mean_fav(row['published'], row['domain'])), axis = 1)
train = train.sort_values('published')
y = train['favs_lognorm'] - train['favs_meanlog']

def extract_features(data_train, data_valid):
    #contentLengthVectorizer = OneHotEncoder()
    max_len_log = log(data_train['content_length'].max())
    #X_train_textlen = contentLengthVectorizer.fit_transform(data_train['content_length']\
    #    .apply(lambda x: int(log1p(x) / max_len_log)).values.reshape(len(data_train), 1))
    #X_valid_textlen = contentLengthVectorizer.transform(data_valid['content_length']\
    #    .apply(lambda x: int(log1p(x) / max_len_log)).values.reshape(len(data_valid), 1))

    title_tfidf = TfidfVectorizer(stop_words=get_stop_words('russian'), analyzer='word', ngram_range=(1, 2))
    X_train_title = title_tfidf.fit_transform(data_train['title'])
    X_valid_title = title_tfidf.transform(data_valid['title'])
    hub_vect = DictVectorizer()
    X_train_hub = hub_vect.fit_transform(data_train['hubs'])
    X_valid_hub = hub_vect.transform(data_valid['hubs'])
    other_dict = DictVectorizer()
    X_train_other = other_dict.fit_transform(data_train[['author', 'domain']].T.to_dict().values())
    X_valid_other = other_dict.transform(data_valid[['author', 'domain']].T.to_dict().values())
    #publ_hour = DictVectorizer()
    #X_train_hour = publ_hour.fit_transform([{time.hour:True} for time in data_train['published']])
    #X_valid_hour = publ_hour.transform([{time.hour:True} for time in data_valid['published']])
    publ_weekday = DictVectorizer()
    X_train_weekday = publ_weekday.fit_transform([{time.weekday():True} for time in data_train['published']])
    X_valid_weekday = publ_weekday.transform([{time.weekday():True} for time in data_valid['published']])
    tags = DictVectorizer()
    X_train_tags = tags.fit_transform([dict((t, True) for t in tags) for tags in data_train['tags']])
    X_valid_tags = tags.transform([dict((t, True) for t in tags) for tags in data_valid['tags']])
    html_tag_regexp = re.compile('<.*?>')
    #content_tfidf = HashingVectorizer(stop_words=get_stop_words('russian'), ngram_range=(1, 2), n_features = 2**18)
    default_prerpocessor = TfidfVectorizer().build_preprocessor()
    remove_html_tags_preprocessor = lambda s: default_prerpocessor(html_tag_regexp.sub('', s))
    content_tfidf = TfidfVectorizer(stop_words=get_stop_words('russian'), analyzer='word', ngram_range=(1, 1), preprocessor = remove_html_tags_preprocessor)
    X_train_content = content_tfidf.fit_transform(data_train['content'])
    X_valid_content = content_tfidf.transform(data_valid['content'])
    X_train_textlen = coo_matrix(data_train['content_length'].apply(lambda x: log1p(x) / max_len_log)).T
    X_valid_textlen = coo_matrix(data_valid['content_length'].apply(lambda x: log1p(x) / max_len_log)).T
    sites = TfidfVectorizer()
    X_train_sites = sites.fit_transform(data_train['sites'].apply(lambda l: ' '.join(s.replace('.', '_') for s in l)))
    X_valid_sites = sites.transform(data_valid['sites'].apply(lambda l: ' '.join(s.replace('.', '_') for s in l)))
    X_train = scipy.sparse.hstack([X_train_title, X_train_other, X_train_content, X_train_weekday, X_train_tags, X_train_textlen, X_train_sites]).tocsr(copy = False) #X_train_hub, 
    X_valid = scipy.sparse.hstack([X_valid_title, X_valid_other, X_valid_content,                               X_valid_weekday, X_valid_tags, X_valid_textlen, X_valid_sites]).tocsr(copy = False) #X_valid_hub,
    return X_train, X_valid

print('Приготовим данные для обучения модели')

X, X_test = extract_features(train, test)

print('Обучим модель')
rgs = linear_model.SGDRegressor(n_iter = 20,  penalty = 'elasticnet', loss = 'squared_epsilon_insensitive', alpha = 0.00001, epsilon = 0.01)
rgs.fit(X, y)

y_test_pred = rgs.predict(X_test)

def get_pred_mean_fav(timestamp, domain):
    return (habr_mean_fav if domain == 'habrahabr.ru' else gt_mean_fav).loc[timestamp.date(), 'favs_mean_pred'] 

test['favs_meanlog'] = test.apply(lambda row: log1p(get_pred_mean_fav(row['published'], row['domain'])), axis = 1)
test['favs_lognorm'] = y_test_pred + test['favs_meanlog']
test[['_id', 'favs_lognorm']].to_csv("my_submission.csv", index = False)

