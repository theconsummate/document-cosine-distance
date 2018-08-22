# -*- coding: utf-8 -*-
from gensim.models.keyedvectors import KeyedVectors
from DocSim import DocSim

# convert the original glove embeddings into word2vec format using this command:
# python3 -m gensim.scripts.glove2word2vec --input data/glove.6B.50d.txt --output data/word2vec.glove.6B.50d.txt
model_path = './data/word2vec.glove.6B.50d.txt'
stopwords_path = "./data/stopwords_en.txt"

model = KeyedVectors.load_word2vec_format(model_path, binary=False)
with open(stopwords_path, 'r') as fh:
    stopwords = fh.read().split(",")
ds = DocSim(model,stopwords=stopwords)

source_doc = "how to delete an invoice"
target_docs = ['delete a invoice', 'how do i remove an invoice', "purge an invoice"]

sim_scores = ds.calculate_similarity(source_doc, target_docs)

print(sim_scores)

# Prints:
##   [ {'score': 0.99999994, 'doc': 'delete a invoice'},
##   {'score': 0.79869318, 'doc': 'how do i remove an invoice'},
##   {'score': 0.71488398, 'doc': 'purge an invoice'} ]
