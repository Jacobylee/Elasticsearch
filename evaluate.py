import heapq

import json
import numpy as np
from scipy.special import softmax
import re
import shelve
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import argparse

from elasticsearch import Elasticsearch

from embedding_service.client import EmbeddingClient
from nltk.corpus import stopwords
from metrics import Score
from text_processing import TextProcessing
from utils import parse_wapo_topics
es = Elasticsearch()


def search(topic_id, index, k, vector, q):
    doc_ids = []
    if vector == "sbert_vector":
        encoder = EmbeddingClient(host="localhost", embedding_type="sbert")
    else:
        encoder = EmbeddingClient(host="localhost", embedding_type="fasttext")
    query_vector = encoder.encode([q], pooling="mean").tolist()[0]
    # get result
    c_result = es.search(index=index, size=40, body={"query": {"match": {"content": q}}})       # todo k
    t_result = es.search(index=index, size=40, body={
                              "query": {
                                "bool": {
                                  "must": [
                                    {
                                      "match": {
                                        "title": {
                                            "query": q,
                                            "boost": 0.5
                                        }
                                      }
                                    }
                                  ]
                                }
                              },
                              "explain": True
                            })         # todo k
    doc_list = {}
    for doc in c_result['hits']['hits']+t_result['hits']['hits']:
        print(doc)
        if vector == "sbert_vector":
            embed_vec = doc['_source']['sbert_vector']
        else:
            embed_vec = doc['_source']['ft_vector']
        cs = cosine_similarity(query_vector, embed_vec)
        doc_list[doc['_id']] = cs
    ordered_doc = sorted(doc_list.items(), key=lambda kv: (kv[1], kv[0]))
    ordered_doc.reverse()
    for i in ordered_doc[:20]:
        print(es.get(index=index, id=i[0], doc_type="_all")['_source']['content'])
        if es.get(index=index, id=i[0], doc_type="_all")['_source']['annotation'].split('-')[0] == topic_id:
            doc_ids.append(int(es.get(index=index, id=i[0], doc_type="_all")['_source']['annotation'].split('-')[1]))
        else:
            doc_ids.append(0)
    print(doc_ids)
    return doc_ids


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_name", required=True, type=str, help="name of the ES index")
    parser.add_argument("--topic_id", required=True, type=int, help="topic id")
    parser.add_argument("--query_type", required=True, type=str, choices=["title", "narration", "description"],
                        help="title, narration, description")
    parser.add_argument('--usermode', '-u', required=False, action='store_true', help='custom content')
    parser.add_argument("--vector_name", required=False, type=str, choices=["sbert_vector", "ft_vector"],
                        help="sbert_vector or ft_vector")
    parser.add_argument("--top_k", required=True, type=int, help="top k")
    return parser.parse_args()


def dot(l1, l2):
    return sum(a*b for a, b in zip(l1, l2))


def cosine_similarity(a, b):
    return dot(a, b) / ((dot(a, a) ** .5) * (dot(b, b) ** .5))


def corpus_filter(corpus):
    # filter used for delete nonalpha char then return filtered corpus
    return [" ".join([re.sub('[^a-z]', '', word.lower()) for word in file.split() if word not in stopwords.words("english")]) for file in corpus]


def tokenize(text):
    return [PorterStemmer().stem(item) for item in nltk.word_tokenize(text)]


def build_corpus(wapo_jl_path):
    with open(wapo_jl_path, "r+", encoding="utf8") as f:
        contents = []
        ids = []
        for item in f:
            obj = json.loads(item)
            contents.append(obj["content_str"])
            ids.append(obj["doc_id"])
    return contents, ids


def tfidf_model(corpus):
    tfidf = TfidfVectorizer(tokenizer=tokenize)
    tfidf.fit_transform(corpus)
    return tfidf

# corpus, ids = build_corpus("pa5_data/test_corpus.jl")
# corpus = corpus_filter(corpus)
# tfidf = tfidf_model(corpus)


def doc_embedding(k, tfidf, doc):
    # return top largest k terms
    response = tfidf.transform([doc])
    feature_names = tfidf.get_feature_names()
    tfidf_dict = {}
    for col in response.nonzero()[1]:
        tfidf_dict[feature_names[col]] = response[0, col]
    h = []
    for value in tfidf_dict:
        heapq.heappush(h, (tfidf_dict[value], value))
    kw = np.array([i[1] for i in heapq.nlargest(k, h)])
    kw_tfidf = np.array([tfidf_dict[word] for word in kw])
    softmaxed = softmax(kw_tfidf)
    encoder = EmbeddingClient(host="localhost", embedding_type="fasttext")
    kwords = " ".join(kw)
    return encoder.encode([kwords], pooling="mean").tolist()[0]


def helper(index):
    result = es.search(index=index, size=40, body={"query": {"match": {"content": "q"}}})       # todo k
    print(result)
    # return query


helper(690)


if __name__ == "__main__":
    query_type_index = {"title": 0, "description": 1, "narration": 2}
    args = build_args()
    # query = parse_wapo_topics("pa5_data/topics2018.xml")[str(args.topic_id)][query_type_index[args.query_type]]
    query0 = parse_wapo_topics("pa5_data/topics2018.xml")[str(args.topic_id)][0]
    query1 = parse_wapo_topics("pa5_data/topics2018.xml")[str(args.topic_id)][1]
    query2 = parse_wapo_topics("pa5_data/topics2018.xml")[str(args.topic_id)][2]
    searched_result0 = search(str(args.topic_id), args.index_name, args.top_k, args.vector_name, query0)
    searched_result1 = search(str(args.topic_id), args.index_name, args.top_k, args.vector_name, query1)
    searched_result2 = search(str(args.topic_id), args.index_name, args.top_k, args.vector_name, query2)
    score = Score
    print(score.eval(searched_result0, args.top_k))
    print(score.eval(searched_result1, args.top_k))
    print(score.eval(searched_result2, args.top_k))


# BERT+default analyzer	0.5	0.3869	0.3333
