import pandas as pd
import joblib
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,precision_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# datasets that is going to evaluate against
names = ["main","article","kaggle"]
datasets = ["article","kaggle"]

# models_name
models_string = ["DT","NB","RF","MLP"]

# BOW-methods 
bows = ["base","base_tfidf","word2vec","word2vec_tfidf"]

# metrics dictionary
acc_metrics = {'base':[[] for i in range (4)],
               'base_tfidf':[[] for i in range (4)],
               'word2vec':[[] for i in range (4)], 
               'word2vec_tfidf':[[] for i in range (4)],
                }
f1_metrics = {'base':[[] for i in range (4)],
              'base_tfidf':[[] for i in range (4)],
              'word2vec':[[] for i in range (4)], 
              'word2vec_tfidf':[[] for i in range (4)],
               }    
precision_metrics = {'base':[[] for i in range (4)],
              'base_tfidf':[[] for i in range (4)],
              'word2vec':[[] for i in range (4)], 
              'word2vec_tfidf':[[] for i in range (4)],
               }   
recall_metrics = {'base':[[] for i in range (4)],
              'base_tfidf':[[] for i in range (4)],
              'word2vec':[[] for i in range (4)], 
              'word2vec_tfidf':[[] for i in range (4)],
               }             

for bow in bows:

    # loading all the model for each BOW-methods
    DT_model = joblib.load('../Model/Tuned/DT_{}.joblib'.format(bow))
    NB_model = joblib.load('../Model/Tuned/NB_{}.joblib'.format(bow))
    RF_model = joblib.load('../Model/Tuned/RF_{}.joblib'.format(bow))
    MLP_model = joblib.load('../Model/Tuned/MLP_{}.joblib'.format(bow))
    print (DT_model.best_params_)
    print (NB_model.best_params_)
    print (RF_model.best_params_)
    print (MLP_model.best_params_)
    
    # loadn and prepare main_dataset from ../Processing/Cleaning.py
    df = pd.read_csv("../Data/Processed/BOW_{}.csv".format(bow))
    label = df['category_label'].tolist()
    X = df.drop(columns=['category_label'])

    X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.15, random_state=1000000)

######################################################################################################################################################
######################################################################################################################################################

    # DT fit, predict and metric calculation
    y_pred = DT_model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred,average="macro")
    precision = precision_score(y_test,y_pred,average="macro")
    recall = recall_score(y_test,y_pred,average="macro")
    print ("DT_{}_{}".format("main",bow))
    print (confusion_matrix(y_test,y_pred))

    acc_metrics[bow][0].append(accuracy)
    f1_metrics[bow][0].append(f1)
    precision_metrics[bow][0].append(precision)
    recall_metrics[bow][0].append(recall)

    ######################################################################################################################################################

    # NB fit, predict and metric calculation
    y_pred = NB_model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred,average="macro")
    precision = precision_score(y_test,y_pred,average="macro")
    recall = recall_score(y_test,y_pred,average="macro")
    print ("NB_{}_{}".format("main",bow))
    print (confusion_matrix(y_test,y_pred))

    acc_metrics[bow][1].append(accuracy)
    f1_metrics[bow][1].append(f1)
    precision_metrics[bow][1].append(precision)
    recall_metrics[bow][1].append(recall)

    ######################################################################################################################################################

    # RF fit, predict and metric calculation
    y_pred = RF_model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred,average="macro")
    precision = precision_score(y_test,y_pred,average="macro")
    recall = recall_score(y_test,y_pred,average="macro")
    print ("RF_{}_{}".format("main",bow))
    print (confusion_matrix(y_test,y_pred))

    acc_metrics[bow][2].append(accuracy)
    f1_metrics[bow][2].append(f1)
    precision_metrics[bow][2].append(precision)
    recall_metrics[bow][2].append(recall)

    ######################################################################################################################################################

    # MLP fit, predict and metric calculation
    y_pred = MLP_model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred,average="macro")
    precision = precision_score(y_test,y_pred,average="macro")
    recall = recall_score(y_test,y_pred,average="macro")
    print ("MLP_{}_{}".format("main",bow))
    print (confusion_matrix(y_test,y_pred))

    acc_metrics[bow][3].append(accuracy)
    f1_metrics[bow][3].append(f1)
    precision_metrics[bow][3].append(precision)
    recall_metrics[bow][3].append(recall)

    # against evaluation dataset (kaggle(https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive) and article)
    for dataset in datasets:

        # data load and preparation
        df = pd.read_csv("../Data/Processed/Eval/BOW_{}_{}.csv".format(dataset,bow))
        df = shuffle(df)
        y_test = df['category_label'].tolist()
        X_test = df.drop(columns=['category_label'])

######################################################################################################################################################
######################################################################################################################################################

        # DT fit, predict and metric calculation
        y_pred = DT_model.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred,average="macro")
        precision = precision_score(y_test,y_pred,average="macro")
        recall = recall_score(y_test,y_pred,average="macro")
        print ("DT_{}_{}".format(dataset,bow))
        print (confusion_matrix(y_test,y_pred))

        acc_metrics[bow][0].append(accuracy)
        f1_metrics[bow][0].append(f1)
        precision_metrics[bow][0].append(precision)
        recall_metrics[bow][0].append(recall)

        ######################################################################################################################################################
        
        # DT fit, predict and metric calculation
        y_pred = NB_model.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred,average="macro")
        precision = precision_score(y_test,y_pred,average="macro")
        recall = recall_score(y_test,y_pred,average="macro")
        print ("NB_{}_{}".format(dataset,bow))
        print (confusion_matrix(y_test,y_pred))

        acc_metrics[bow][1].append(accuracy)
        f1_metrics[bow][1].append(f1)
        precision_metrics[bow][1].append(precision)
        recall_metrics[bow][1].append(recall)

        ######################################################################################################################################################
        
        # DT fit, predict and metric calculation
        y_pred = RF_model.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred,average="macro")
        precision = precision_score(y_test,y_pred,average="macro")
        recall = recall_score(y_test,y_pred,average="macro")
        print ("RF_{}_{}".format(dataset,bow))
        print (confusion_matrix(y_test,y_pred))

        acc_metrics[bow][2].append(accuracy)
        f1_metrics[bow][2].append(f1)
        precision_metrics[bow][2].append(precision)
        recall_metrics[bow][2].append(recall)

        ######################################################################################################################################################

        # DT fit, predict and metric calculation
        y_pred = MLP_model.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred,average="macro")
        precision = precision_score(y_test,y_pred,average="macro")
        recall = recall_score(y_test,y_pred,average="macro")
        print ("MLP_{}_{}".format(dataset,bow))
        print (confusion_matrix(y_test,y_pred))

        acc_metrics[bow][3].append(accuracy)
        f1_metrics[bow][3].append(f1)
        precision_metrics[bow][3].append(precision)
        recall_metrics[bow][3].append(recall)

# print (acc_metrics)
# print (f1_metrics)
# print (precision_metrics)
# print (recall_metrics)


# PRINTING METRICS 

print ("""
####################################################################################################
Accuracy
####################################################################################################
""")
for i in range (3):
    print ("""
###########################################
# {}        """.format(names[i]))

    print ("""                       
######### #################################  
# model # #  base # b_idf #  w2v  # w_idf #  
######### ################################# """)
    for j in range(4):
        print ("# {:^5} # # {:<5.3f} # {:<5.3f} # {:<5.3f} # {:<5.3f} #"\
                .format(models_string[j]\
                ,acc_metrics['base'][j][i],acc_metrics['base_tfidf'][j][i],acc_metrics['word2vec'][j][i],acc_metrics['word2vec_tfidf'][j][i]))

    print ("######### #################################")

#####################################################################################################################

print ("""
####################################################################################################
F1
####################################################################################################
""")
for i in range (3):
    print ("""
###########################################
# {}        """.format(names[i]))

    print ("""                       
######### #################################  
# model # #  base # b_idf #  w2v  # w_idf #  
######### ################################# """)
    for j in range(4):
        print ("# {:^5} # # {:<5.3f} # {:<5.3f} # {:<5.3f} # {:<5.3f} #"\
                .format(models_string[j]\
                ,f1_metrics['base'][j][i],f1_metrics['base_tfidf'][j][i],f1_metrics['word2vec'][j][i],f1_metrics['word2vec_tfidf'][j][i]))

    print ("######### #################################")

#####################################################################################################################

print ("""
####################################################################################################
Precision
####################################################################################################
""")
for i in range (3):
    print ("""
###########################################
# {}        """.format(names[i]))

    print ("""                       
######### #################################  
# model # #  base # b_idf #  w2v  # w_idf #  
######### ################################# """)
    for j in range(4):
        print ("# {:^5} # # {:<5.3f} # {:<5.3f} # {:<5.3f} # {:<5.3f} #"\
                .format(models_string[j]\
                ,precision_metrics['base'][j][i],precision_metrics['base_tfidf'][j][i],precision_metrics['word2vec'][j][i],precision_metrics['word2vec_tfidf'][j][i]))

    print ("######### #################################")

#####################################################################################################################

print ("""
####################################################################################################
Recall
####################################################################################################
""")
for i in range (3):
    print ("""
###########################################
# {}        """.format(names[i]))

    print ("""                       
######### #################################  
# model # #  base # b_idf #  w2v  # w_idf #  
######### ################################# """)
    for j in range(4):
        print ("# {:^5} # # {:<5.3f} # {:<5.3f} # {:<5.3f} # {:<5.3f} #"\
                .format(models_string[j]\
                ,recall_metrics['base'][j][i],recall_metrics['base_tfidf'][j][i],recall_metrics['word2vec'][j][i],recall_metrics['word2vec_tfidf'][j][i]))

    print ("######### #################################")