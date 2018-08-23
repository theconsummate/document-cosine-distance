### Install the dependencies
Use python 3
```
pip3 install -r requirements.txt
```

### Download glove embeddings
#### English
Url: https://nlp.stanford.edu/projects/glove/
Url for smallest 6 billion corpus: http://nlp.stanford.edu/data/glove.6B.zip

Extract the zip file and copy the glove.6B.50d.txt file into the data folder.

### Convert the glove file into word2vec format
```
python3 -m gensim.scripts.glove2word2vec --input data/glove.6B.50d.txt --output data/word2vec.glove.6B.50d.txt
```

#### Italian
Download from http://hlt.isti.cnr.it/wordembeddings/glove_wiki_window10_size300_iteration50.tar.gz and then copy the extracted files into data folder

### Create GoogleAPI keys
#### Create an API key
Go to https://developers.google.com/custom-search/json-api/v1/overview and 'Get a Key' button in the API key section. Follow the steps along.

#### Custom Search engine key
Go to https://cse.google.com/cse/ and create a key. In the 'sites to search field', add

```
*.com
*.co.uk
*.it
```

Next, create a file keys.py in this directory and put your keys here
```
developer_key=<Developer_key>
search_engine_key=<Custom Search engine key>
```

### Compute the cosine-distance
```
python3 main <lang_code> "<query string>"
```

Retrieves the top 10 search results for the input query string from Google and computes the distance of the first result with the other 9 results.

Comparison is done only for snippets and not the entire webpage of the search result.
