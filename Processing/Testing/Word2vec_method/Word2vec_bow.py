import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# to change the category to numerical class
category_dict = {'health':0,'politics':1,'science':2,'business':3,'technology':4,'sport':5}

# candidate of number of feature
num_features = [5000,10000,15000,20000]

# load the data for method 1 word2vec
df_1 = pd.read_csv('../../../Data/Testing/Word2vec/Data_word2vec_v1.csv')
for num_feature in num_features:
    # initializing countvectorier to generate bag-of-word
    vectorizer = CountVectorizer(ngram_range=(1,2),max_features=num_feature)

    ##########################################################################################################################################

    # data preparation
    contents_1 = df_1['content']
    categories_1 = df_1['category'].tolist()

    #########################################################################################################################################
    ##########################################################################################################################################

    # create and save the bag of word
    X = vectorizer.fit_transform(contents_1)
    X = X.toarray()
    X = np.transpose(X)
    X = np.append(X,[categories_1],axis=0)
    X = np.transpose(X)

    columns = vectorizer.get_feature_names()+['category_label']
    df_bow = pd.DataFrame(X,columns=columns)
    df_bow.replace({'category_label':category_dict},inplace=True)
    df_bow.to_csv('../../../Data/Testing/Word2vec/BOW_word2vec_v1_{}.csv'.format(num_feature),index=False) 
    
    ##########################################################################################################################################

    # load the data for method 2 word2vec
    df_2 = pd.read_csv("../../../Data/Testing/Word2vec/Data_word2vec_v2_{}.csv".format(num_feature))

    ##########################################################################################################################################

    # data preparation
    contents_2 = df_2['content']
    categories_2 = df_2['category'].tolist()

    #########################################################################################################################################
    ##########################################################################################################################################

    # create and save the bag of word
    X = vectorizer.fit_transform(contents_2)
    X = X.toarray()
    X = np.transpose(X)
    X = np.append(X,[categories_2],axis=0)
    X = np.transpose(X)

    columns = vectorizer.get_feature_names()+['category_label']
    df_bow = pd.DataFrame(X,columns=columns)
    df_bow.replace({'category_label':category_dict},inplace=True)
    df_bow.to_csv('../../../Data/Testing/Word2vec/BOW_word2vec_v2_{}.csv'.format(num_feature),index=False) 
