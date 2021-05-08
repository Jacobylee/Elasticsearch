import argparse

from elasticsearch import Elasticsearch

from embedding_service.client import EmbeddingClient
from metrics import Score
from utils import parse_wapo_topics
es = Elasticsearch()


def search(topic_id, index, k, custom, vector, q):
    doc_ids = []
    if not vector:
        if custom:
            result = es.search(index=index, size=k, body={"query": {"match": {"custom_content": q}}})
        else:
            result = es.search(index=index, size=k, body={"query": {"match": {"content": q}}})
        for doc in result['hits']['hits']:
            if doc['_source']['annotation'].split('-')[0] == topic_id:
                doc_ids.append(int(doc['_source']['annotation'].split('-')[1]))
            else:
                doc_ids.append(0)
    elif not custom:
        if vector == "sbert_vector":
            encoder = EmbeddingClient(host="localhost", embedding_type="sbert")
        else:
            encoder = EmbeddingClient(host="localhost", embedding_type="fasttext")
        query_vector = encoder.encode([q], pooling="mean").tolist()[0]
        # get result
        result = es.search(index=index, size=k, body={"query": {"match": {"content": q}}})
        doc_list = {}
        for doc in result['hits']['hits']:
            if vector == "sbert_vector":
                embed_vec = doc['_source']['sbert_vector']
            else:
                embed_vec = doc['_source']['ft_vector']
            cs = cosine_similarity(query_vector, embed_vec)
            doc_list[doc['_id']] = cs
        ordered_doc = sorted(doc_list.items(), key=lambda kv: (kv[1], kv[0]))
        ordered_doc.reverse()
        for i in ordered_doc:
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


if __name__ == "__main__":
    query_type_index = {"title": 0, "description": 1, "narration": 2}
    args = build_args()
    query = parse_wapo_topics("pa5_data/topics2018.xml")[str(args.topic_id)][query_type_index[args.query_type]]
    searched_result = search(str(args.topic_id), args.index_name, args.top_k, args.usermode, args.vector_name, query)
    score = Score
    print(score.eval(searched_result, args.top_k))
    # q = 2
    # query0 = parse_wapo_topics("pa5_data/topics2018.xml")["321"][q]
    # searched_result0 = search('321', args.index_name, args.top_k, args.usermode, args.vector_name, query0)
    #
    # query1 = parse_wapo_topics("pa5_data/topics2018.xml")["336"][q]
    # searched_result1 = search('336', args.index_name, args.top_k, args.usermode, args.vector_name, query1)
    #
    # query2 = parse_wapo_topics("pa5_data/topics2018.xml")["341"][q]
    # searched_result2 = search('341', args.index_name, args.top_k, args.usermode, args.vector_name, query2)
    #
    # query3 = parse_wapo_topics("pa5_data/topics2018.xml")["347"][q]
    # searched_result3 = search('347', args.index_name, args.top_k, args.usermode, args.vector_name, query3)
    #
    # query4 = parse_wapo_topics("pa5_data/topics2018.xml")["397"][q]
    # searched_result4 = search('397', args.index_name, args.top_k, args.usermode, args.vector_name, query4)
    #
    # score = Score
    # print(score.eval(searched_result0, args.top_k))
    # print(score.eval(searched_result1, args.top_k))
    # print(score.eval(searched_result2, args.top_k))
    # print(score.eval(searched_result3, args.top_k))
    # print(score.eval(searched_result4, args.top_k))
