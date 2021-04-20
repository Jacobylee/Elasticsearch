from elasticsearch import Elasticsearch
from flask import Flask, render_template, request
from embedding_service.client import EmbeddingClient
app = Flask(__name__)
es = Elasticsearch()
docs = []
query_fix = ""
index_file = "wapo_docs_50k"


# home page
@app.route("/")
def home():
    return render_template("home.html")


# result page
@app.route("/results", methods=["POST"])
def results():  # todo custom not working
    query = request.form["query"]  # Get the raw user query from home page
    global query_fix
    query_fix = query
    searchway = request.form["searchway"]
    global docs
    docs.clear()
    doc_ids = []
    doc_ids.clear()
    # apply search way
    if searchway == "bm25deault":   # BM25 + default analyzer
        result = es.search(index=index_file, size=20, body={"query": {"match": {"content": query}}})
        doc_ids = []
        for doc in result['hits']['hits']:
            doc_ids.append(doc['_id'])
    elif searchway == "bm25custom":    # BM25 + custom analyzer
        result = es.search(index=index_file, size=20, body={"query": {"match": {"custom_content": query}}})
        doc_ids = []
        for doc in result['hits']['hits']:
            doc_ids.append(doc['_id'])
    else:
        if searchway == "fastText":    # Bert + default analyzer
            encoder = EmbeddingClient(host="localhost", embedding_type="fasttext")
        else:   # fastText + default analyzer
            encoder = EmbeddingClient(host="localhost", embedding_type="sbert")
        query_vector = encoder.encode([query], pooling="mean").tolist()[0]
        # get result
        result = es.search(index=index_file, size=20, body={"query": {"match": {"content": query}}})
        doc_list = {}
        for doc in result['hits']['hits']:
            if searchway == "fastText": # Bert + default analyzer
                embed_vec = doc['_source']['ft_vector']
            else:   # fastText + default analyzer
                embed_vec = doc['_source']['sbert_vector']
            cs = cosine_similarity(query_vector, embed_vec)
            doc_list[doc['_id']] = cs
        ordered_doc = sorted(doc_list.items(), key=lambda kv: (kv[1], kv[0]))
        ordered_doc.reverse()
        doc_ids = [i[0] for i in ordered_doc]
    # get result
    for d in doc_ids:
        doc = {}
        doc['id'] = d
        doc['title'] = es.get(index=index_file, id=d, doc_type="_all")['_source']['title']
        doc['snippet'] = es.get(index=index_file, id=d, doc_type="_all")['_source']['content'][:150]
        docs.append(doc)
    # judge eight each page
    if len(docs) <= 8:
        return render_template("results.html", lst=docs, page=1, total=len(docs), last=True, query=query)
    else:
        return render_template("results.html", lst=docs[:8], page=1, total=len(docs), query=query)


# "next page" to show more results
@app.route("/results/<int:page_id>", methods=["GET"])
def next_page(page_id):
    if page_id * 8 < len(docs):
        return render_template("results.html", lst=docs[(page_id - 1) * 8:page_id * 8], page=page_id,
                               total=len(docs), query=query_fix)
    else:
        return render_template("results.html", lst=docs[(page_id - 1) * 8:len(docs)], page=page_id,
                               total=len(docs), last=True, query=query_fix)


# document page
@app.route("/doc_data/<int:doc_id>")
def doc_data(doc_id):
    # get information
    target = es.get(index=index_file, id=str(doc_id), doc_type="_all")
    doc = target['_source']
    title = doc['title']
    author = doc['author']
    main = doc["content"]
    date = doc["date"]
    annotation = doc["annotation"]
    return render_template("doc.html", title=title, author=author, date=date, main=main, annotation=annotation)


def dot(l1, l2):
    return sum(a*b for a, b in zip(l1, l2))


def cosine_similarity(a, b):
    return dot(a, b) / ((dot(a, a) ** .5) * (dot(b, b) ** .5))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
