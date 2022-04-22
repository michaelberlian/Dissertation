from gensim.models import Word2Vec,KeyedVectors


############################################################ 1
# testing word2vec model that built using dataset from Cleaning.py
windows = [4,6,8,10,12]
min_counts = [1,2,3]
vector_sizes = [50,100,150,200]
for vector_size in vector_sizes:
    for window in windows:
        for min_count in min_counts:
            for sg in [0,1]:
                model = Word2Vec.load('../../../Model/Testing/Word2vec/word2vec_{}_{}_{}_{}.model'.format(vector_size,window,min_count,sg))
                print ("########################")
                print ("vectorsize_{}_window_{}_mincount_{}_skipgram_{}".format(vector_size,window,min_count,sg))
                print(model.wv.most_similar("democracy")[:5])
                print(model.wv.similarity(w1="democracy", w2="america"))
                print(model.wv.most_similar("covid")[:5])
                print(model.wv.similarity(w1="vaccine", w2="covid"))
                print(model.wv.most_similar("computer")[:5])
                print(model.wv.similarity(w1="technology", w2="science"))
                print(model.wv.most_similar("olympic")[:5])
                print(model.wv.similarity(w1="medal", w2="olympic"))
            
############################################################ 2

# testing word2vec model that built using dataset "text8" from gensim api 
model = Word2Vec.load('../../../Model/Testing/Word2vec/word2vec_text8.model')
print ("######################## word2vec_text8")
print(model.wv.most_similar("democracy")[:5])
print(model.wv.similarity(w1="democracy", w2="america"))
# print(model.wv.most_similar("covid")[:5]) token is not exist in this model
# print(model.wv.similarity(w1="vaccine", w2="covid")) token is not exist in this model
print(model.wv.most_similar("computer")[:5])
print(model.wv.similarity(w1="technology", w2="science"))
print(model.wv.most_similar("olympic")[:5])
print(model.wv.similarity(w1="medal", w2="olympic"))

# testing word2vec model that built using dataset of "text8" and Cleaning.py from gensim api 
model = Word2Vec.load('../../../Model/Testing/Word2vec/word2vec_text8_added.model')
print ("######################## word2vec_text8_added")
print(model.wv.most_similar("democracy")[:5])
print(model.wv.similarity(w1="democracy", w2="america"))
print(model.wv.most_similar("covid")[:5])
print(model.wv.similarity(w1="vaccine", w2="covid"))
print(model.wv.most_similar("computer")[:5])
print(model.wv.similarity(w1="technology", w2="science"))
print(model.wv.most_similar("olympic")[:5])
print(model.wv.similarity(w1="medal", w2="olympic"))

#############################################################

# testing pre-traing model "word2vec_word2vec-google-news-300"
model = KeyedVectors.load('../../../Model/Testing/Word2vec/word2vec_word2vec-google-news-300.model')
print ("######################## word2vec_word2vec-google-news-300")
print(model.most_similar("democracy")[:5])
print(model.similarity(w1="democracy", w2="america"))
# print(model.most_similar("covid")[:5]) token is not exist in this model
# print(model.similarity(w1="vaccine", w2="covid")) token is not exist in this model
print(model.most_similar("computer")[:5])
print(model.similarity(w1="technology", w2="science"))
print(model.most_similar("olympic")[:5])
print(model.similarity(w1="medal", w2="olympic"))

# testing pre-traing model "word2vec_word2vec-google-news-300"
model = Word2Vec.load('../../../Model/Testing/Word2vec/word2vec_20-newsgroups.model')
print ("######################## word2vec_20-newsgroups")
# print(model.wv.most_similar("democracy")[:5]) token is not exist in this model
# print(model.wv.similarity(w1="democracy", w2="america")) token is not exist in this model
# print(model.wv.most_similar("health")[:5]) token is not exist in this model
# print(model.wv.similarity(w1="vaccine", w2="quarantine")) token is not exist in this model
# print(model.wv.most_similar("computer")[:5]) token is not exist in this model
# print(model.wv.similarity(w1="technology", w2="science")) token is not exist in this model
# print(model.wv.most_similar("olympic")[:5]) token is not exist in this model
# print(model.wv.similarity(w1="medal", w2="olympic")) token is not exist in this model

#############################################################

# testing word2vec model that built using dataset simpsons dataset from kaggle : https://www.kaggle.com/code/pierremegret/gensim-word2vec-tutorial/data
model = Word2Vec.load('../../../Model/Testing/Word2vec/word2vec_simpsons.model')
print ("######################## word2vec_simpsons")
print(model.wv.most_similar("democracy")[:5])
print(model.wv.similarity(w1="democracy", w2="america"))
print(model.wv.most_similar("health")[:5])
print(model.wv.similarity(w1="vaccine", w2="quarantine"))
print(model.wv.most_similar("computer")[:5])
print(model.wv.similarity(w1="technology", w2="science"))
print(model.wv.most_similar("olympic")[:5])
print(model.wv.similarity(w1="medal", w2="olympic"))
