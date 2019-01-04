# -*- coding: utf-8 -*-
from googleapiclient.discovery import build
import pprint
import sys
import keys
import csv
import codecs

# lda
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(400)
import nltk
nltk.download('wordnet')


pp = pprint.PrettyPrinter()


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
    # print(query)
    dictionary_results = query['items']

    docs = []
    for i in range(len(dictionary_results)):
        result = dictionary_results[i]
        doc = {"id": i, "content": result["snippet"], "link": result["link"], "title": result["title"]}
        docs.append(doc)
    return docs


def lemmatize_stemming(text, stemmer):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# Tokenize and lemmatize
def preprocess(text, stemmer):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token, stemmer))

    return result

def find_topics(documents, language):
    if language == "en":
        stemmer = SnowballStemmer("english")
    elif language == "it":
        stemmer = SnowballStemmer("italian")
    else:
        print("language not supported")
        sys.exit()

    processed_docs = []

    for doc in documents:
        processed_docs.append(preprocess(doc, stemmer))

    '''
    Create a dictionary from 'processed_docs' containing the number of times a word appears
    in the training set using gensim.corpora.Dictionary and call it 'dictionary'
    '''
    dictionary = gensim.corpora.Dictionary(processed_docs)

    '''
    Create the Bag-of-words model for each document i.e for each document we create a dictionary reporting how many
    words and how many times those words appear. Save this to 'bow_corpus'
    '''

    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    '''
    Train your lda model using gensim.models.LdaMulticore and save it to 'lda_model'
    '''
    lda_model =  gensim.models.LdaMulticore(bow_corpus,
                                       num_topics = 8,
                                       id2word = dictionary,
                                       passes = 10,
                                       workers = 2)
    topic_words = []
    # print("topics found in the search results.")
    for idx, topic in lda_model.print_topics(-1):
        print("Topic: {} \nWords: {}".format(idx, topic ))
        topic_words += re.findall('"([^"]*)"', topic)
        # print("\n")
    return set(topic_words)


def keyword_planner(keywordfile, language):
    contents = csv.reader(codecs.open(keywordfile, 'rU', 'utf-16'), delimiter='\t')
    keywords = [row[0] for row in contents]
    return find_topics(keywords, language)



if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("""Please provide an input query string to get search results. Run like this:
python3 topics.py <language_code> "<query string>"
possible values for language_code: en or it""")
        sys.exit()
    language = sys.argv[1]
    query_str = sys.argv[2]
    docs = get_search_results(query_str, language)
    snippets = [doc['content'] for doc in docs]
    print("search results ...")
    pp.pprint(docs)
    # keys:content, id, link, title
    # print(documents)
    print("topics in search results ...")
    search_topics = find_topics(snippets, language)
    print("topics in keywords ...")
    keyword_topics = keyword_planner('keywords.csv', language)
    print("topics in keywords which are not present in search results ...")
    print(keyword_topics - search_topics)

