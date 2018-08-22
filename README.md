### Install the dependencies
Use python 3
```
pip3 install -r requirements.txt
```

### Download glove embeddings
Url: https://nlp.stanford.edu/projects/glove/
Url for smallest 6 billion corpus: http://nlp.stanford.edu/data/glove.6B.zip

Extract the zip file and copy the glove.6B.50d.txt file into the data folder.

### Convert the glove file into word2vec format
```
python3 -m gensim.scripts.glove2word2vec --input data/glove.6B.50d.txt --output data/word2vec.glove.6B.50d.txt
```

### Compute the cosine-distance
```
python3 main "<query string>"
```

Retrieves the top 10 search results for the input query string from Google and computes the distance of the first result with the other 9 results.
