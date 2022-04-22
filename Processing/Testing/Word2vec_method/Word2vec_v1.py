import pandas as pd 
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

# to remove puctuation from words
def remove_punctuation(punctuations,content):
    spaces = ' '*len(punctuations)
    table = content.maketrans(punctuations,spaces)
    return content.translate(table)

# load data and model
df = pd.read_csv('../../../Data/Processed/Data.csv')
model = Word2Vec.load('../../../Model/Word2vec/word2vec.model')

contents = df['content'].tolist()
punctuation = ".?\""

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
        similar_words = model.wv.most_similar(word)
        # find top 5 most similar word of current word with similarity more than 0.95 (on scale 0-1) and check if the available similar word is exist on features
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

# print(len(features))
# print(len(vocab))

new_contents = []

# applying the dictionary to the content
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

# same the data with altered content
df['content'] = new_contents
df.to_csv('../../../Data/Testing/Word2vec/Data_word2vec_v1.csv')

######################################################################

