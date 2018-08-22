# -*- coding: utf-8 -*-
from gensim.models.keyedvectors import KeyedVectors
from DocSim import DocSim
from lib.google_search_results import GoogleSearchResults
import pprint
import sys

model_path = './data/word2vec.glove.6B.50d.txt'
stopwords_path = "./data/stopwords_en.txt"
pp = pprint.PrettyPrinter()

def query_word2vec(source_doc, target_docs):
    model = KeyedVectors.load_word2vec_format(model_path, binary=False)
    with open(stopwords_path, 'r') as fh:
        stopwords = fh.read().split(",")
    ds = DocSim(model,stopwords=stopwords)

    sim_scores = ds.calculate_similarity(source_doc, target_docs)

    print(sim_scores)


def get_search_results(query_str):
    params = {
        "q" : query_str,
        # "location" : "Austin, Texas, United States",
        "hl" : "en",
        "google_domain" : "google.com",
        "api_key" : "demo",
        "safe" : "active",
        "num" : "10",
    }

    query = GoogleSearchResults(params)
    dictionary_results = query.get_dictionary()['organic_results']

    docs = []
    for result in dictionary_results:
        doc = {"id": result["position"], "content": result["snippet"], "link": result["link"], "title": result["title"]}
        docs.append(doc)
    return docs

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("""Please provide an input query string to get search results. Run like this:
python3 main.py "<query string>" """)
        sys.exit()
    query_str = sys.argv[1]
    docs = get_search_results(query_str)
    pp.pprint(docs)
    print("computing similarity ...")
    query_word2vec(docs[0]["content"], docs[1:])

