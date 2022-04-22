import pandas as pd 
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer

# model load
model = Word2Vec.load('../../../Model/Word2vec/word2vec.model')

# candidate of Number of features
num_features = [5000,10000,15000,20000]

for num_feature in num_features:

    # load data 
    df = pd.read_csv('../../../Data/Processed/Data.csv')
    # print(df)
    contents = df['content']
    new_contents = []

    # create the bag-of-word features list 
    vectorizer = CountVectorizer(ngram_range=(1,2),max_features=num_feature)
    X = vectorizer.fit_transform(contents)
    features = vectorizer.get_feature_names()

    # print (len(features))
    # looped through content
    for content in contents:
        # print(content)
        words = word_tokenize(content)
        new_content = ""
        # looped through each word in content
        for word in words:
            found = False
            # if the word token is in the bag-of-word feature list then no changes
            if word in features:
                new_content += word + ' '
                # print(1, word, features.index(word))
            # if the word token is NOT in the bag-of-word feature list 
            else :
                try :
                    # try to find a similar word with the current word that exist in the bag-of-word feature list
                    similar_words = model.wv.most_similar(word)
                    for j in range (5):
                        # if found then alter the word to the similar word
                        if similar_words[j][1] > 0.95 and similar_words[j][0] in features:
                            new_content += similar_words[j][0] + ' '
                            found = True
                            # print('found', word, similar_words[j][0])
                            break
                    # if not found then no changes
                    if not found:
                        new_content += word + ' '  
                        # print("not found", word, word)
                # if not found then no changes  
                except : 
                    new_content += word + ' '
                    # print ("not found", word, word)
        # print (new_content)
        new_contents.append(new_content)

    #save the altered data
    df['content'] = new_contents
    df.to_csv('../../../Data/Testing/Word2vec/Data_word2vec_v2_{}.csv'.format(num_feature))