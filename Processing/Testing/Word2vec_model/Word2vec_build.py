import pandas as pd 
import gensim
import numpy as np
from nltk.tokenize import sent_tokenize
from gensim.models import Word2Vec

# load data that had been cleaned and random sampled from ../Cleaning.py
df = pd.read_csv('../../../Data/Processed/Data.csv')

# create a list of sentence from list of content
content_list = df['content'].tolist()
arr = [] 
counter = 0 
for content in content_list:
    try:
        sentences = sent_tokenize(content)
        arr += sentences
    except:
        counter += 1

# dropping duplicates and use simple preprocess to the sentences provided by gensim
df = pd.DataFrame([arr],index=['content'],columns=np.arange(len(arr)))
df = df.transpose()
df.drop_duplicates(inplace=True)
contents = df['content'].apply(gensim.utils.simple_preprocess)
# print (contents)
# contents = df['content']

# create Word2vec models from gensim library with multiple parameters combination 
windows = [4,6,8,10,12]
min_counts = [1,2,3]
vector_sizes = [50,100,150,200]
for vector_size in vector_sizes:
    for window in windows:
        for min_count in min_counts:
            for sg in [0,1]:
                model = Word2Vec(
                    vector_size=vector_size,
                    window=window,
                    min_count=min_count,
                    workers=4,
                    sg=1,
                )
                model.build_vocab(contents)
                model.train(contents, total_examples=model.corpus_count, epochs=model.epochs)
                # save the model to be tested on Word2vec_compare.py
                model.save("../../../Model/Testing/Word2vec/word2vec_{}_{}_{}_{}.model".format(vector_size,window,min_count,sg))
