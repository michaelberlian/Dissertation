import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import numpy as np

# to change the category to numerical class
category_dict = {'health':0,'politics':1,'science':2,'business':3,'technology':4,'sport':5}

# num of features candidates
# num_features = [5000,10000,15000,20000,None]
num_features = [5000,10000,15000,20000]

# load and prepare the data
df_base = pd.read_csv('../../../Data/Processed/Data.csv')
contents_base = df_base['content']
categories_base = df_base['category'].tolist()

# load and prepare the data
df_word2vec = pd.read_csv('../../..//Data/Processed/Data_word2vec.csv',index_col=False)
contents_word2vec = df_word2vec['content']
categories_word2vec = df_word2vec['category'].tolist()

# repeat process for each candidate
for num_feature in num_features:
    # initialize vectorizers
    vectorizer = CountVectorizer(ngram_range=(1,2),max_features=num_feature)
    tf_vectorizer = TfidfVectorizer(ngram_range=(1,2),max_features=num_feature)

##########################################################################################################################################
    
    # create BOW_base and save it
    X = vectorizer.fit_transform(contents_base)
    X = X.toarray()
    X = np.transpose(X)
    X = np.append(X,[categories_base],axis=0)
    X = np.transpose(X)

    columns = vectorizer.get_feature_names()+['category_label']
    df_bow = pd.DataFrame(X,columns=columns)
    df_bow.replace({'category_label':category_dict},inplace=True)
    df_bow.to_csv('../../../Data/Testing/NumFeatures/BOW_base_{}.csv'.format(str(num_feature)),index=False)

# ##########################################################################################################################################

    # create BOW_Word2vec and save it
    X = vectorizer.fit_transform(contents_word2vec)
    X = X.toarray()
    X = np.transpose(X)
    X = np.append(X,[categories_word2vec],axis=0)
    X = np.transpose(X)

    columns = vectorizer.get_feature_names()+['category_label']
    df_bow = pd.DataFrame(X,columns=columns)
    df_bow.replace({'category_label':category_dict},inplace=True)
    df_bow.to_csv('../../../Data/Testing/NumFeatures/BOW_word2vec_{}.csv'.format(str(num_feature)),index=False)

##########################################################################################################################################

    # create BOW_base_tfidf and save it
    X = tf_vectorizer.fit_transform(contents_base)
    X = X.toarray()
    X = np.transpose(X)
    X = np.append(X,[categories_base],axis=0)
    X = np.transpose(X)

    columns = tf_vectorizer.get_feature_names()+['category_label']
    df_bow = pd.DataFrame(X,columns=columns)
    df_bow.replace({'category_label':category_dict},inplace=True)
    df_bow.to_csv('../../../Data/Testing/NumFeatures/BOW_base_tfidf_{}.csv'.format(str(num_feature)),index=False)

##########################################################################################################################################

    # create BOW_word2vec_tdidf and save it
    X = tf_vectorizer.fit_transform(contents_word2vec)
    X = X.toarray()
    X = np.transpose(X)
    X = np.append(X,[categories_word2vec],axis=0)
    X = np.transpose(X)

    columns = tf_vectorizer.get_feature_names()+['category_label']
    df_bow = pd.DataFrame(X,columns=columns)
    df_bow.replace({'category_label':category_dict},inplace=True)
    df_bow.to_csv('../../../Data/Testing/NumFeatures/BOW_word2vec_tfidf_{}.csv'.format(str(num_feature)),index=False)
