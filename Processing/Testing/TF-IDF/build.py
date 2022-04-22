import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# to change the category to numerical class
category_dict = {'health':0,'politics':1,'science':2,'business':3,'technology':4,'sport':5}

# candidate of number of feature
num_features = [5000,10000,15000,20000]

# load and prepare the data 
df = pd.read_csv('../../../Data/Processed/Data.csv')
contents = df['content']
categories = df['category'].tolist()

for num_feature in num_features:

    #create bag of word using word that appear the most 
    tf_vectorizer = TfidfVectorizer(ngram_range=(1,2), use_idf=True, max_features=num_feature)

##########################################################################################################################################

    X = tf_vectorizer.fit_transform(contents)
    X = X.toarray()
    X = np.transpose(X)
    X = np.append(X,[categories],axis=0)
    X = np.transpose(X)

    columns = tf_vectorizer.get_feature_names()+['category_label']
    df_bow = pd.DataFrame(X,columns=columns)
    df_bow.replace({'category_label':category_dict},inplace=True)
    df_bow.to_csv('../../../Data/Testing/TF-IDF/BOW_base_tf-idf_most_{}.csv'.format(num_feature),index=False)

##########################################################################################################################################
##########################################################################################################################################

    # getting words that are appear the least by using idf 
    tf_vectorizer = TfidfVectorizer(ngram_range=(1,2), use_idf=True)

    X = tf_vectorizer.fit_transform(contents)

    features = tf_vectorizer.get_feature_names()
    idf = tf_vectorizer.idf_

    features = np.array(features)
    index = idf.argsort()
    indexes = index[-num_feature:]
    print (len(indexes))
    features_cut = features[indexes]
##########################################################################################################################################

    # create bag of word using vocabulary that appear the least 
    tf_vectorizer = TfidfVectorizer(ngram_range=(1,2), use_idf=True, max_features=num_feature, vocabulary=features_cut)

    X = tf_vectorizer.fit_transform(contents)
    X = X.toarray()
    X = np.transpose(X)
    X = np.append(X,[categories],axis=0)
    X = np.transpose(X)

    columns = tf_vectorizer.get_feature_names()+['category_label']
    df_bow = pd.DataFrame(X,columns=columns)
    df_bow.replace({'category_label':category_dict},inplace=True)
    df_bow.to_csv('../../../Data/Testing/TF-IDF/BOW_base_tf-idf_least_{}.csv'.format(num_feature),index=False)

    # print(df_bow)
    # print(vectorizer.get_feature_names())
    # print(len(vectorizer.get_feature_names()))