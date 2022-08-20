from multiprocessing.resource_sharer import stop
from traceback import print_tb
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
df = pd.read_csv('data/dataset.csv')
description_list = df['Description']
stopwords_list = df['Keywords']
combined_list = df['Combined']

def tfidf_recom(idx):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(description_list)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    sim_scores_tfidf = list(enumerate(cosine_sim[idx]))
    return sim_scores_tfidf

def sw_recom(idx):
    sw_count = CountVectorizer(stop_words='english')
    sw_matrix = sw_count.fit_transform(stopwords_list)
    cosine_sim_sw = cosine_similarity(sw_matrix, sw_matrix)
    sim_scores_sw = list(enumerate(cosine_sim_sw[idx]))
    return sim_scores_sw

def comb_recom(idx):
    comb = TfidfVectorizer(stop_words='english')
    comb_matrix = comb.fit_transform(combined_list)
    cosine_sim_comb = linear_kernel(comb_matrix, comb_matrix)
    sim_scores_comb = list(enumerate(cosine_sim_comb[idx]))
    return sim_scores_comb

def DescriptionReco(title, recom_type):
    idx = df[df.Title == title].index[0]
    description = df.iloc[idx]['Description']
    keywords = df.iloc[idx]['Keywords']
    combined = title + ' ' + keywords

    sim_scores = recom_type(idx)[1:6]
    res_indices = [i[0] for i in sim_scores]
    titles = df['Title'].iloc[res_indices]
    return titles.tolist()

# print(DescriptionReco('ConceptLearner: Discovering Visual Concepts from Weakly Labeled Image Collections', tfidf_recom))
# print(DescriptionReco('ConceptLearner: Discovering Visual Concepts from Weakly Labeled Image Collections', sw_recom))
print(DescriptionReco('Refactoring: Improving the Design of Existing Code', comb_recom))