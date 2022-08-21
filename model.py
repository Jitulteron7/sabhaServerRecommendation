import enum
from operator import itemgetter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

df = pd.read_csv('data/dataset.csv')
title_list = df['Title'].tolist()
description_list = df['Description'].tolist()
keywords_list = df['Keywords'].tolist()

def get_tfidf(title, num = 5):
    idx = title_list.index(title)
    description = description_list[idx]

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(description_list)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)
    sim_scores = sim_scores[1:num+1]
    res_indices = [i[0] for i in sim_scores]
    titles = list((itemgetter(*res_indices)(title_list)))
    return titles

def get_kw(title, num = 5):
    idx = title_list.index(title)
    keywords = keywords_list[idx]

    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df['Keywords'])
    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

    sim_scores = list(enumerate(cosine_sim2[idx]))
    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)
    sim_scores = sim_scores[1:num+1]
    res_indices = [i[0] for i in sim_scores]
    titles = list((itemgetter(*res_indices)(title_list)))
    return titles

print(get_tfidf('Building an FPS Game with Unity',3))
print(get_kw('Building an FPS Game with Unity',3))