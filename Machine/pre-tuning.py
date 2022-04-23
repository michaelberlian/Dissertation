from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import time

# BOW-methods
bows = ["base","base_tfidf","word2vec","word2vec_tfidf"]
bows_name = ["base","b_tfidf","w2v","w_tfidf"]

# models-name
models_string = ["DT","NB","RF","MLP"]
counter = 0 

# dictionary to store metrics
acc_metrics = {'base':[],
               'base_tfidf':[],
               'word2vec':[], 
               'word2vec_tfidf':[],
                }
f1_metrics = {'base':[],
              'base_tfidf':[],
              'word2vec':[], 
              'word2vec_tfidf':[],
               }    
precision_metrics = {'base':[],
              'base_tfidf':[],
              'word2vec':[], 
              'word2vec_tfidf':[],
               }   
recall_metrics = {'base':[],
              'base_tfidf':[],
              'word2vec':[], 
              'word2vec_tfidf':[],
               }    
          
for i in range (len(bows)):

    # load and prepare data
    df = pd.read_csv("../Data/Processed/BOW_{}.csv".format(bows[i]))

    label = df['category_label'].tolist()
    X = df.drop(columns=['category_label'])
    X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.15, random_state=1000000)
    
    # DT fit, predict and metric calculation
    DT = DecisionTreeClassifier()
    model = DT.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred,average="macro")
    precision = precision_score(y_test,y_pred,average="macro")
    recall = recall_score(y_test,y_pred,average="macro")
    acc_metrics[bows[i]].append(accuracy)
    f1_metrics[bows[i]].append(f1)
    precision_metrics[bows[i]].append(precision)
    recall_metrics[bows[i]].append(recall)
    
    # NB fit, predict and metric calculation
    NB = GaussianNB()
    model = NB.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred,average="macro")
    precision = precision_score(y_test,y_pred,average="macro")
    recall = recall_score(y_test,y_pred,average="macro")
    acc_metrics[bows[i]].append(accuracy)
    f1_metrics[bows[i]].append(f1)
    precision_metrics[bows[i]].append(precision)
    recall_metrics[bows[i]].append(recall)

    # RF fit, predict and metric calculation
    RF = RandomForestClassifier()
    model = RF.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred,average="macro")
    precision = precision_score(y_test,y_pred,average="macro")
    recall = recall_score(y_test,y_pred,average="macro")
    acc_metrics[bows[i]].append(accuracy)
    f1_metrics[bows[i]].append(f1)
    precision_metrics[bows[i]].append(precision)
    recall_metrics[bows[i]].append(recall)

    # MLP fit, predict and metric calculation
    MLP = MLPClassifier()
    model = MLP.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred,average="macro")
    precision = precision_score(y_test,y_pred,average="macro")
    recall = recall_score(y_test,y_pred,average="macro")
    acc_metrics[bows[i]].append(accuracy)
    f1_metrics[bows[i]].append(f1)
    precision_metrics[bows[i]].append(precision)
    recall_metrics[bows[i]].append(recall)

    print ("{} DONE".format(bows[i]))


# PRINTING METRICS
print ("""
########################################################################
accuracy 

          accuracy                            
######### #################################  
# model # #  base # b_idf #  w2v  # w_idf #  
######### #################################  """)
for i in range (4):
    print ("""# {:^5} # # {:<5.3f} # {:<5.3f} # {:<5.3f} # {:<5.3f} #"""\
    .format(models_string[i]\
        ,acc_metrics['base'][i],acc_metrics['base_tfidf'][i],acc_metrics['word2vec'][i],acc_metrics['word2vec_tfidf'][i]))

print ("######### #################################")

print ("""
########################################################################
f1 

          f1                            
######### #################################  
# model # #  base # b_idf #  w2v  # w_idf #  
######### #################################  """)
for i in range (4):
    print ("""# {:^5} # # {:<5.3f} # {:<5.3f} # {:<5.3f} # {:<5.3f} #"""\
    .format(models_string[i]\
        ,f1_metrics['base'][i],f1_metrics['base_tfidf'][i],f1_metrics['word2vec'][i],f1_metrics['word2vec_tfidf'][i]))

print ("######### #################################")

print ("""
########################################################################
preccision 

          precision                            
######### #################################  
# model # #  base # b_idf #  w2v  # w_idf #  
######### #################################  """)
for i in range (4):
    print ("""# {:^5} # # {:<5.3f} # {:<5.3f} # {:<5.3f} # {:<5.3f} #"""\
    .format(models_string[i]\
        ,precision_metrics['base'][i],precision_metrics['base_tfidf'][i],precision_metrics['word2vec'][i],precision_metrics['word2vec_tfidf'][i]))

print ("######### #################################")

print ("""
########################################################################
recall 

          recall                            
######### #################################  
# model # #  base # b_idf #  w2v  # w_idf #  
######### #################################  """)
for i in range (4):
    print ("""# {:^5} # # {:<5.3f} # {:<5.3f} # {:<5.3f} # {:<5.3f} #"""\
    .format(models_string[i]\
        ,recall_metrics['base'][i],recall_metrics['base_tfidf'][i],recall_metrics['word2vec'][i],recall_metrics['word2vec_tfidf'][i]))

print ("######### #################################")
