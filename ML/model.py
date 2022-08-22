import enum
from operator import itemgetter
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

# df = pd.read_csv('data\dataset.csv')
# title_list = df['Title'].tolist()
# description_list = df['Description'].tolist()
# keywords_list = df['Keywords'].tolist()

f = open("data/data-js.json")
json_file = json.load(f)

def get_data(json_file):
    desc_list = []
    kw_list = []
    fileIdx_list = []
    for i in range(len(json_file)):
        fileIdx = json_file[i]['fileIdx']
        desc = json_file[i]['desc']
        kw = ' '.join(json_file[i]['keyWords'])
        desc_list.append(desc)
        fileIdx_list.append(fileIdx)
        kw_list.append(kw)
    return kw_list, desc_list, fileIdx_list

def get_tfidf(fileIdx, num = 5):
    kw_list, desc_list, fileIdx_list = get_data(json_file)
    for i in range(len(fileIdx_list)):
        if fileIdx == fileIdx_list[i]:
            idx = i
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(desc_list)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)
    sim_scores = sim_scores[1:num+1]
    res_indices = [i[0] for i in sim_scores]
    res_indices = list((itemgetter(*res_indices)(fileIdx_list)))
    return res_indices

def get_kw(fileIdx, num = 5):
    kw_list, desc_list, fileIdx_list = get_data(json_file)
    for i in range(len(fileIdx_list)):
        if fileIdx == fileIdx_list[i]:
            idx = i
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(kw_list)
    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

    sim_scores = list(enumerate(cosine_sim2[idx]))
    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)
    sim_scores = sim_scores[1:num+1]
    res_indices = [i[0] for i in sim_scores]
    res_indices = list((itemgetter(*res_indices)(fileIdx_list)))
    return res_indices

print(get_tfidf(12,5))
print(get_kw(12,5))