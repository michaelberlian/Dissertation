import pandas as pd 
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import pickle

# remove punctuation from the word
def remove_punctuation(punctuations,content):
    spaces = ' '*len(punctuations)
    table = content.maketrans(punctuations,spaces)
    return content.translate(table)

# load the data and model
df = pd.read_csv('../Data/Processed/Data.csv')
model = Word2Vec.load('../Model/Word2vec/word2vec.model')

contents = df['content'].tolist()
punctuation = ".?"

# create a single list of all tokens of words
word_tokens = []
for content in contents:
    words = word_tokenize(content)
    words = [remove_punctuation(punctuation,word) for word in words]
    word_tokens += words

# remove duplicates 
word_tokens = list(set(word_tokens))

# print (word_tokens)
# print (len(word_tokens))

# features : collection of word is considered 
# vocab : used to alter the content of the dataset 
# vocab['token'] = "considered word in features | token"
features = []
vocab = {}
# creating a dictionary that assigned each token of word into a similar token in features or its own if no other similar token exist in features
for word in word_tokens:
    found = False
    close_words = []
    try:
        # find top 5 most similar word of current word with similarity more than 0.95 (on scale 0-1) and check if the available similar word is exist on features
        similar_words = model.wv.most_similar(word)
        for j in range (5):
            if similar_words[j][1] > 0.95 and similar_words[j][0] in features:
                vocab[word] = similar_words[j][0]
                found = True
                break
        if not found:
            # if the word is the first of its kind or no similar word in considered words then it is considered
            features.append(word)
            vocab[word] = word
    except:
        # words that are not found in word2vec model instantly considered
        features.append(word)
        vocab[word] = word

print(len(features))
print(len(vocab))

# save the features and vocab for later use 
pickle_out = open("../Data/Pickles/Word2vec_dictionary.pickle","wb")
pickle.dump(vocab, pickle_out)
pickle_out.close()

pickle_out = open("../Data/Pickles/Word2vec_words.pickle","wb")
pickle.dump(features, pickle_out)
pickle_out.close()

######################################################################
# print ("data #######################################")
# applying the dictionary to the content of dataset from Cleaning.py
df = pd.read_csv('../Data/Processed/Data.csv')
contents = df['content'].tolist()

new_contents = []

for content in contents:
    # print(content)
    words = word_tokenize(content)
    new_content = ""
    for word in words:
        try:
            new_content += vocab[word] + ' '
            # print(1, word, vocab[word])
        except:
            new_content += word + ' '
            # print (2, word, word)
    # print (new_content)
    new_contents.append(new_content)

df['content'] = new_contents
df.to_csv('../Data/Processed/Data_word2vec.csv')

######################################################################
# print ("article #######################################")

# applying the dictionary to the content of article dataset
df_article = pd.read_csv('../Data/Processed/Eval/Data_article.csv')
contents = df_article['content'].tolist()

new_contents = []

for content in contents:
    # print(content)
    words = word_tokenize(content)
    new_content = ""
    for word in words:
        try:
            new_content += vocab[word] + ' '
            # print(1, word, vocab[word])
        except:
            new_content += word + ' '
            # print (2, word, word)
    # print (new_content)
    new_contents.append(new_content)

df_article['content'] = new_contents
df_article.to_csv('../Data/Processed/Eval/Data_article_word2vec.csv')

######################################################################
# print ("kaggle #######################################")

# applying the dictionary to the content of dataset from kaggle : https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive
df_kaggle = pd.read_csv('../Data/Processed/Eval/Data_kaggle.csv')
contents = df_kaggle['content'].tolist()

new_contents = []

for content in contents:
    # print(content)
    words = word_tokenize(content)
    new_content = ""
    for word in words:
        try:
            new_content += vocab[word] + ' '
            # print(1, word, vocab[word])
        except:
            new_content += word + ' '
            # print (2, word, word)
    # print (new_content)
    new_contents.append(new_content)

df_kaggle['content'] = new_contents
df_kaggle.to_csv('../Data/Processed/Eval/Data_kaggle_word2vec.csv')

