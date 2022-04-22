from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

#Initializing lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

#loading BBC NEWS
df = pd.read_excel('../Data/Raw/BBC_news.xlsx')
contents = df['content'].tolist()
new_contents = []

for content in contents:
    # print(content)
    words = word_tokenize(content)
    new_content = ""
    for word in words:
        # do not include the word if it is stop word or only number with less than 4 character
        if (word not in stop_words) and not (word.isnumeric() and len(word) < 4):
            new_content += lemmatizer.lemmatize(word) + ' '
    new_contents.append(new_content.lower())

#replace with lemmatized content and save
df['content'] = new_contents
df.to_csv('../Data/Cleaned/BBC_news_lemmatized.csv')

###################################################################################################

#loading CNN NEWS
df = pd.read_excel('../Data/Raw/CNN_news.xlsx')
contents = df['content'].tolist()
new_contents = []

for content in contents:
    # print(content)
    words = word_tokenize(content)
    new_content = ""
    for word in words:
        # do not include the word if it is stop word or only number with less than 4 character
        if (word not in stop_words) and not (word.isnumeric() and len(word) < 4):
            new_content += lemmatizer.lemmatize(word) + ' '
    new_contents.append(new_content.lower())

#replace with lemmatized content and save
df['content'] = new_contents
df.to_csv('../Data/Cleaned/CNN_news_lemmatized.csv')

###################################################################################################

#loading journal article
df = pd.read_excel('../Data/Raw/article.xlsx')
contents = df['content'].tolist()
new_contents = []

lemmatizer = WordNetLemmatizer()

for content in contents:
    # print(content)
    words = word_tokenize(content)
    new_content = ""
    for word in words:
        # do not include the word if it is stop word or only number with less than 4 character
        if (word not in stop_words) and not (word.isnumeric() and len(word) < 4):
            new_content += lemmatizer.lemmatize(word) + ' '
    new_contents.append(new_content.lower())

#replace with lemmatized content and save
df['content'] = new_contents
df.to_csv('../Data/Cleaned/article_lemmatized.csv')

###################################################################################################

#loading news dataset from kaggle : https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive
df = pd.read_csv('../Data/Raw/bbc-kaggle.csv',sep="\t")
contents = df['content'].tolist()
new_contents = []

lemmatizer = WordNetLemmatizer()

for content in contents:
    # print(content)
    words = word_tokenize(content)
    new_content = ""
    for word in words:
        # do not include the word if it is stop word or only number with less than 4 character
        if (word not in stop_words) and not (word.isnumeric() and len(word) < 4):
            new_content += lemmatizer.lemmatize(word) + ' '
    new_contents.append(new_content.lower())

#replace with lemmatized content and save
df['content'] = new_contents
#remove news with category entertainment because it is not part of our category
df = df[df['category'] != 'entertainment']
#rename the category tech to technology to be persistent with our dataset
df.replace({'category':'tech'},'technology',inplace=True)
df.to_csv('../Data/Cleaned/kaggle_lemmatized.csv')
df.to_csv('../Data/Processed/Eval/Data_kaggle.csv')

#################################
#################################

#combining the BBC, CNN, and part of journal article as training set

#rest of journal article will be test set
df_BBC = pd.read_csv('../Data/Cleaned/BBC_news_lemmatized.csv')
df_CNN = pd.read_csv('../Data/Cleaned/CNN_news_lemmatized.csv')
df_article = pd.read_csv('../Data/Cleaned/article_lemmatized.csv')
#take 35 of each category to be train set from journal article
df_article_train = df_article.groupby("category").sample(n=35, random_state=20402764)
#removing the train set from the test set
df_article_test = pd.concat([df_article, df_article_train]).drop_duplicates(keep=False)

df = pd.concat([df_BBC,df_CNN])
#take 350 of each category to be train set from BBC and CNN
df = df.groupby("category").sample(n=350, random_state=20402764)

#combining the BBC, CNN, and journal article
df = pd.concat([df,df_article_train])

#save
df.to_csv('../Data/Processed/Data.csv',index=False)
df_article_test.to_csv('../Data/Processed/Eval/Data_article.csv')

