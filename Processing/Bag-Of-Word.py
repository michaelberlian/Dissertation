import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import numpy as np
import pickle

# to change the category to numerical class
category_dict = {'health':0,'politics':1,'science':2,'business':3,'technology':4,'sport':5}

# load and prepare the data
df_base = pd.read_csv('../Data/Processed/Data.csv')
contents_base = df_base['content']
categories_base = df_base['category'].tolist()

# load and prepare the data
df_word2vec = pd.read_csv('../Data/Processed/Data_word2vec.csv',index_col=False)
contents_word2vec = df_word2vec['content']
categories_word2vec = df_word2vec['category'].tolist()

# base-20000
##########################################################################################################################################

# create BOW_base and save it
vectorizer = CountVectorizer(ngram_range=(1,2),max_features=20000)
X = vectorizer.fit_transform(contents_base)
X = X.toarray()
# adding columns of category
X = np.transpose(X)
X = np.append(X,[categories_base],axis=0)
X = np.transpose(X)

columns = vectorizer.get_feature_names()+['category_label'] # adding the label the the row
df_bow = pd.DataFrame(X,columns=columns)
df_bow.replace({'category_label':category_dict},inplace=True) # applying the category dict to make numerical category class
df_bow.to_csv('../Data/Processed/BOW_base.csv',index=False)

pickle_out = open("../Data/Pickles/base_feature.pickle","wb")
pickle.dump(vectorizer.get_feature_names(), pickle_out)
pickle_out.close()

# word2vec-15000
##########################################################################################################################################

# create BOW_word2vec and save it
vectorizer = CountVectorizer(ngram_range=(1,2),max_features=15000)
X = vectorizer.fit_transform(contents_word2vec)
X = X.toarray()
X = np.transpose(X)
X = np.append(X,[categories_word2vec],axis=0)
X = np.transpose(X)

columns = vectorizer.get_feature_names()+['category_label']
df_bow = pd.DataFrame(X,columns=columns)
df_bow.replace({'category_label':category_dict},inplace=True)
df_bow.to_csv('../Data/Processed/BOW_word2vec.csv',index=False)

pickle_out = open("../Data/Pickles/word2vec_feature.pickle","wb")
pickle.dump(vectorizer.get_feature_names(), pickle_out)
pickle_out.close()

#tfidf
##########################################################################################################################################

#base_tfidf-10000
##########################################################################################################################################

# create BOW_base_tfidf and save it
tf_vectorizer = TfidfVectorizer(ngram_range=(1,2),max_features=10000)
X = tf_vectorizer.fit_transform(contents_base)
X = X.toarray()
X = np.transpose(X)
X = np.append(X,[categories_base],axis=0)
X = np.transpose(X)

columns = tf_vectorizer.get_feature_names()+['category_label']
df_bow = pd.DataFrame(X,columns=columns)
df_bow.replace({'category_label':category_dict},inplace=True)
df_bow.to_csv('../Data/Processed/BOW_base_tfidf.csv',index=False)

pickle_out = open("../Data/Pickles/base_tfidf_feature.pickle","wb")
pickle.dump(tf_vectorizer.get_feature_names(), pickle_out)
pickle_out.close()

#word2vec_tfidf-15000
##########################################################################################################################################

# create BOW_word2vec_tfidf and save it
tf_vectorizer = TfidfVectorizer(ngram_range=(1,2),max_features=15000)
X = tf_vectorizer.fit_transform(contents_word2vec)
X = X.toarray()
X = np.transpose(X)
X = np.append(X,[categories_word2vec],axis=0)
X = np.transpose(X)

columns = tf_vectorizer.get_feature_names()+['category_label']
df_bow = pd.DataFrame(X,columns=columns)
df_bow.replace({'category_label':category_dict},inplace=True)
df_bow.to_csv('../Data/Processed/BOW_word2vec_tfidf.csv',index=False)

pickle_out = open("../Data/Pickles/word2vec_tfidf_feature.pickle","wb")
pickle.dump(tf_vectorizer.get_feature_names(), pickle_out)
pickle_out.close()

##########################################################################################################################################
##########################################################################################################################################

# loading feature list of each Bag-of-Word methods
pickle_in = open("../Data/Pickles/base_feature.pickle", "rb")
base_feature = pickle.load(pickle_in)
pickle_in = open("../Data/Pickles/word2vec_feature.pickle", "rb")
w2v_feature = pickle.load(pickle_in)
pickle_in = open("../Data/Pickles/base_tfidf_feature.pickle", "rb")
base_tfidf_feature = pickle.load(pickle_in)
pickle_in = open("../Data/Pickles/word2vec_tfidf_feature.pickle", "rb")
w2v_tfidf_feature = pickle.load(pickle_in)
print (len(base_feature), len(w2v_feature), len(base_tfidf_feature), len(w2v_tfidf_feature))

#Article
##########################################################################################################################################

# load and prepare the data 
df_base = pd.read_csv('../Data/Processed/Eval/Data_article.csv')
contents_base = df_base['content']
categories_base = df_base['category'].tolist()

df_word2vec = pd.read_csv('../Data/Processed/Eval/Data_article_word2vec.csv',index_col=False)
contents_word2vec = df_word2vec['content']
categories_word2vec = df_word2vec['category'].tolist()

# base-20000
##########################################################################################################################################

# create BOW_base using feature list from train dataset and save it
vectorizer = CountVectorizer(ngram_range=(1,2),vocabulary=base_feature)
X = vectorizer.fit_transform(contents_base)
X = X.toarray()
X = np.transpose(X)
X = np.append(X,[categories_base],axis=0)
X = np.transpose(X)

columns = vectorizer.get_feature_names()+['category_label']
df_bow = pd.DataFrame(X,columns=columns)
df_bow.replace({'category_label':category_dict},inplace=True)
df_bow.to_csv('../Data/Processed/Eval/BOW_article_base.csv',index=False)
print (len(df_bow.columns))

# word2vec-15000
##########################################################################################################################################

# create BOW_word2vec using feature list from train dataset and save it
vectorizer = CountVectorizer(ngram_range=(1,2),vocabulary=w2v_feature)
X = vectorizer.fit_transform(contents_word2vec)
X = X.toarray()
X = np.transpose(X)
X = np.append(X,[categories_word2vec],axis=0)
X = np.transpose(X)

columns = vectorizer.get_feature_names()+['category_label']
df_bow = pd.DataFrame(X,columns=columns)
df_bow.replace({'category_label':category_dict},inplace=True)
df_bow.to_csv('../Data/Processed/Eval/BOW_article_word2vec.csv',index=False)
print (len(df_bow.columns))

#tfidf
##########################################################################################################################################

#base_tfidf-10000
##########################################################################################################################################

# create BOW_base_tfidf using feature list from train dataset and save it
tf_vectorizer = TfidfVectorizer(ngram_range=(1,2),vocabulary=base_tfidf_feature)
X = tf_vectorizer.fit_transform(contents_base)
X = X.toarray()
X = np.transpose(X)
X = np.append(X,[categories_base],axis=0)
X = np.transpose(X)

columns = tf_vectorizer.get_feature_names()+['category_label']
df_bow = pd.DataFrame(X,columns=columns)
df_bow.replace({'category_label':category_dict},inplace=True)
df_bow.to_csv('../Data/Processed/Eval/BOW_article_base_tfidf.csv',index=False)
print (len(df_bow.columns))

#word2vec_tfidf-15000
##########################################################################################################################################

# create BOW_word2vec_tfidf using feature list from train dataset and save it
tf_vectorizer = TfidfVectorizer(ngram_range=(1,2),vocabulary=w2v_tfidf_feature)
X = tf_vectorizer.fit_transform(contents_word2vec)
X = X.toarray()
X = np.transpose(X)
X = np.append(X,[categories_word2vec],axis=0)
X = np.transpose(X)

columns = tf_vectorizer.get_feature_names()+['category_label']
df_bow = pd.DataFrame(X,columns=columns)
df_bow.replace({'category_label':category_dict},inplace=True)
df_bow.to_csv('../Data/Processed/Eval/BOW_article_word2vec_tfidf.csv',index=False)
print (len(df_bow.columns))


#Kaggle
##########################################################################################################################################

# load and prepare the data 
df_base = pd.read_csv('../Data/Processed/Eval/Data_kaggle.csv')
contents_base = df_base['content']
categories_base = df_base['category'].tolist()

df_word2vec = pd.read_csv('../Data/Processed/Eval/Data_kaggle_word2vec.csv',index_col=False)
contents_word2vec = df_word2vec['content']
categories_word2vec = df_word2vec['category'].tolist()

# base-20000
##########################################################################################################################################

# create BOW_base using feature list from train dataset and save it
vectorizer = CountVectorizer(ngram_range=(1,2),vocabulary=base_feature)
X = vectorizer.fit_transform(contents_base)
X = X.toarray()
X = np.transpose(X)
X = np.append(X,[categories_base],axis=0)
X = np.transpose(X)

columns = vectorizer.get_feature_names()+['category_label']
df_bow = pd.DataFrame(X,columns=columns)
df_bow.replace({'category_label':category_dict},inplace=True)
df_bow.to_csv('../Data/Processed/Eval/BOW_kaggle_base.csv',index=False)
print (len(df_bow.columns))

# word2vec-15000
##########################################################################################################################################

# create BOW_word2vec using feature list from train dataset and save it
vectorizer = CountVectorizer(ngram_range=(1,2),vocabulary=w2v_feature)
X = vectorizer.fit_transform(contents_word2vec)
X = X.toarray()
X = np.transpose(X)
X = np.append(X,[categories_word2vec],axis=0)
X = np.transpose(X)

columns = vectorizer.get_feature_names()+['category_label']
df_bow = pd.DataFrame(X,columns=columns)
df_bow.replace({'category_label':category_dict},inplace=True)
df_bow.to_csv('../Data/Processed/Eval/BOW_kaggle_word2vec.csv',index=False)
print (len(df_bow.columns))

#tfidf
##########################################################################################################################################

#base_tfidf-10000
##########################################################################################################################################

# create BOW_base_tfidf using feature list from train dataset and save it
tf_vectorizer = TfidfVectorizer(ngram_range=(1,2),vocabulary=base_tfidf_feature)
X = tf_vectorizer.fit_transform(contents_base)
X = X.toarray()
X = np.transpose(X)
X = np.append(X,[categories_base],axis=0)
X = np.transpose(X)

columns = tf_vectorizer.get_feature_names()+['category_label']
df_bow = pd.DataFrame(X,columns=columns)
df_bow.replace({'category_label':category_dict},inplace=True)
df_bow.to_csv('../Data/Processed/Eval/BOW_kaggle_base_tfidf.csv',index=False)
print (len(df_bow.columns))

#word2vec_tfidf-15000
##########################################################################################################################################

# create BOW_word2vec_tfidf using feature list from train dataset and save it
tf_vectorizer = TfidfVectorizer(ngram_range=(1,2),vocabulary=w2v_tfidf_feature)
X = tf_vectorizer.fit_transform(contents_word2vec)
X = X.toarray()
X = np.transpose(X)
X = np.append(X,[categories_word2vec],axis=0)
X = np.transpose(X)

columns = tf_vectorizer.get_feature_names()+['category_label']
df_bow = pd.DataFrame(X,columns=columns)
df_bow.replace({'category_label':category_dict},inplace=True)
df_bow.to_csv('../Data/Processed/Eval/BOW_kaggle_word2vec_tfidf.csv',index=False)
print (len(df_bow.columns))