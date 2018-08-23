# -*- coding: utf-8 -*-
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec
from DocSim import DocSim
from googleapiclient.discovery import build
import pprint
import sys
import keys

model_path_en = './data/word2vec.glove.6B.50d.txt'
model_path_it = './data/glove_WIKI'
stopwords_path_en = "./data/stopwords_en.txt"
stopwords_path_it = "./data/stopwords-it.txt"
pp = pprint.PrettyPrinter()

def load_model(language):
    if language == "en":
        model = KeyedVectors.load_word2vec_format(model_path_en, binary=False)
        with open(stopwords_path_en, 'r') as fh:
            stopwords = fh.read().split(",")
        return model, stopwords

    elif language == "it":
        model = Word2Vec.load(model_path_it)
        with open(stopwords_path_it, 'r') as fh:
            stopwords = [x.strip() for x in fh.readlines()]
        return model, stopwords
    else:
        print("Invalid language code. Only en and it supported")
        sys.exit()

def query_word2vec(source_doc, target_docs, language):
    model, stopwords = load_model(language)
    ds = DocSim(model,stopwords=stopwords)

    sim_scores = ds.calculate_similarity(source_doc, target_docs)

    print(sim_scores)


def get_search_results(query_str, language="en"):
    if language == "en":
        domain = "google.com"
        lr = "lang_en"
    elif language == "it":
        domain = "google.it"
        lr = "lang_it"
    # params = {
    #     "q" : query_str,
    #     # "location" : "Austin, Texas, United States",
    #     "hl" : language,
    #     "google_domain" : domain,
    #     "api_key" : "demo",
    #     "safe" : "active",
    #     "num" : "10",
    # }
    service = build("customsearch", "v1", developerKey=keys.developer_key)
    query = service.cse().list(
                               q=query_str,
                               hl=language,
                               lr=lr,
                               googlehost=domain,
                               cx=keys.search_engine_key,
                               # cx='017576662512468239146:omuauf_lfve',
                               ).execute()
    print(query)
    dictionary_results = query['items']

    docs = []
    for i in range(len(dictionary_results)):
        result = dictionary_results[i]
        doc = {"id": i, "content": result["snippet"], "link": result["link"], "title": result["title"]}
        docs.append(doc)
    return docs

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("""Please provide an input query string to get search results. Run like this:
python3 main.py <language_code> "<query string>"
possible values for language_code: en or it""")
        sys.exit()
    language = sys.argv[1]
    query_str = sys.argv[2]
    docs = get_search_results(query_str, language)
    pp.pprint(docs)
    print("computing similarity ...")
    query_word2vec(docs[0]["content"], docs[1:], language)

