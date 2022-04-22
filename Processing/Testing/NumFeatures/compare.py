from numpy import average,median
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import time

# bows = ["base"]
# num_features = [None,5000]
# prepare the data
bows = ["base","base_tfidf","word2vec","word2vec_tfidf"]
num_features = [None]
# num_features = [5000,10000,15000,20000,None]
# num_features = [5000,10000,15000,20000]
NOF = ["5K","10K","15K","20K","None"]

models_string = ["DT","NB","RF","MLP"]
counter = 0 

acc_metrics = {'base':[[] for i in range (5)],
               'base_tfidf':[[] for i in range (5)],
               'word2vec':[[] for i in range (5)], 
               'word2vec_tfidf':[[] for i in range (5)],
                }
f1_metrics = {'base':[[] for i in range (5)],
              'base_tfidf':[[] for i in range (5)],
              'word2vec':[[] for i in range (5)], 
              'word2vec_tfidf':[[] for i in range (5)],
               }      
time_metrics = {'base':[[] for i in range (5)],
              'base_tfidf':[[] for i in range (5)],
              'word2vec':[[] for i in range (5)], 
              'word2vec_tfidf':[[] for i in range (5)],
               }             

# looped through each type of bow and each number of feature candidate 
for i in range (len(num_features)):
    for j in range (len(bows)):

        # data loading and preparation
        df = pd.read_csv("../../../Data/Testing/NumFeatures/BOW_{}_{}.csv".format(bows[j],str(num_features[i])))

        label = df['category_label'].tolist()
        X = df.drop(columns=['category_label'])
        X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.15, random_state=1000000)
        
        # DT fit, predict and metric calculation
        start_time = time.time()
        DT = DecisionTreeClassifier()
        model = DT.fit(X_train,y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred,average="macro")
        time_metrics[bows[j]][i].append(time.time()-start_time)
        acc_metrics[bows[j]][i].append(accuracy)
        f1_metrics[bows[j]][i].append(f1)
        

        # NB fit, predict and metric calculation
        start_time = time.time()
        NB = GaussianNB()
        model = NB.fit(X_train,y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred,average="macro")
        time_metrics[bows[j]][i].append(time.time()-start_time)
        acc_metrics[bows[j]][i].append(accuracy)
        f1_metrics[bows[j]][i].append(f1)

        # RF fit, predict and metric calculation
        start_time = time.time()
        RF = RandomForestClassifier()
        model = RF.fit(X_train,y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred,average="macro")
        time_metrics[bows[j]][i].append(time.time()-start_time)
        acc_metrics[bows[j]][i].append(accuracy)
        f1_metrics[bows[j]][i].append(f1)

        # MLP fit, predict and metric calculation
        start_time = time.time()
        MLP = MLPClassifier()
        model = MLP.fit(X_train,y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred,average="macro")
        time_metrics[bows[j]][i].append(time.time()-start_time)
        acc_metrics[bows[j]][i].append(accuracy)
        f1_metrics[bows[j]][i].append(f1)

        print ("{}_{} DONE".format(bows[j],len(df.columns)))
        # print (time_metrics)
        # print (acc_metrics)
        # print (f1_metrics)


# PRINTING THE RESULTS

print ("""
########################################################################
base 

        accuracy                                           f1 
####### #################################################  #################################################
# NOF # #   DT  #   NB  #   RF  #  MLP  #  med  #  avg  #  #   DT  #   NB  #   RF  #  MLP  #  med  #  avg  #
####### #################################################  #################################################""")
for j in range (len(num_features)):
    print ("""#{:^5}# # {:<5.3f} # {:<5.3f} # {:<5.3f} # {:<5.3f} # {:<5.3f} # {:<5.3f} #  # {:<5.3f} # {:<5.3f} # {:<5.3f} # {:<5.3f} # {:<5.3f} # {:<5.3f} #"""\
    .format(NOF[j]\
        ,acc_metrics['base'][j][0],acc_metrics['base'][j][1],acc_metrics['base'][j][2],acc_metrics['base'][j][3], median(acc_metrics['base'][j]),average(acc_metrics['base'][j])\
        ,f1_metrics['base'][j][0],f1_metrics['base'][j][1],f1_metrics['base'][j][2],f1_metrics['base'][j][3],median(f1_metrics['base'][j]),average(f1_metrics['base'][j])))

print ("####### #################################################  #################################################")

print ("""
########################################################################
word2vec 

        accuracy                                           f1 
####### #################################################  #################################################
# NOF # #   DT  #   NB  #   RF  #  MLP  #  med  #  avg  #  #   DT  #   NB  #   RF  #  MLP  #  med  #  avg  #
####### #################################################  #################################################""")
for j in range (len(num_features)):
    print ("""#{:^5}# # {:<5.3f} # {:<5.3f} # {:<5.3f} # {:<5.3f} # {:<5.3f} # {:<5.3f} #  # {:<5.3f} # {:<5.3f} # {:<5.3f} # {:<5.3f} # {:<5.3f} # {:<5.3f} #"""\
    .format(NOF[j]\
        ,acc_metrics['word2vec'][j][0],acc_metrics['word2vec'][j][1],acc_metrics['word2vec'][j][2],acc_metrics['word2vec'][j][3], median(acc_metrics['word2vec'][j]),average(acc_metrics['word2vec'][j])\
        ,f1_metrics['word2vec'][j][0],f1_metrics['word2vec'][j][1],f1_metrics['word2vec'][j][2],f1_metrics['word2vec'][j][3],median(f1_metrics['word2vec'][j]),average(f1_metrics['word2vec'][j])))

print ("####### #################################################  #################################################")

print ("""
########################################################################
base_tfidf 

        accuracy                                           f1 
####### #################################################  #################################################
# NOF # #   DT  #   NB  #   RF  #  MLP  #  med  #  avg  #  #   DT  #   NB  #   RF  #  MLP  #  med  #  avg  #
####### #################################################  #################################################""")
for j in range (len(num_features)):
    print ("""#{:^5}# # {:<5.3f} # {:<5.3f} # {:<5.3f} # {:<5.3f} # {:<5.3f} # {:<5.3f} #  # {:<5.3f} # {:<5.3f} # {:<5.3f} # {:<5.3f} # {:<5.3f} # {:<5.3f} #"""\
    .format(NOF[j]\
        ,acc_metrics['base_tfidf'][j][0],acc_metrics['base_tfidf'][j][1],acc_metrics['base_tfidf'][j][2],acc_metrics['base_tfidf'][j][3], median(acc_metrics['base_tfidf'][j]),average(acc_metrics['base_tfidf'][j])\
        ,f1_metrics['base_tfidf'][j][0],f1_metrics['base_tfidf'][j][1],f1_metrics['base_tfidf'][j][2],f1_metrics['base_tfidf'][j][3],median(f1_metrics['base_tfidf'][j]),average(f1_metrics['base_tfidf'][j])))

print ("####### #################################################  #################################################")

print ("""
########################################################################
word2vec_tfidf 

        accuracy                                           f1 
####### #################################################  #################################################
# NOF # #   DT  #   NB  #   RF  #  MLP  #  med  #  avg  #  #   DT  #   NB  #   RF  #  MLP  #  med  #  avg  #
####### #################################################  #################################################""")
for j in range (len(num_features)):
    print ("""#{:^5}# # {:<5.3f} # {:<5.3f} # {:<5.3f} # {:<5.3f} # {:<5.3f} # {:<5.3f} #  # {:<5.3f} # {:<5.3f} # {:<5.3f} # {:<5.3f} # {:<5.3f} # {:<5.3f} #"""\
    .format(NOF[j]\
        ,acc_metrics['word2vec_tfidf'][j][0],acc_metrics['word2vec_tfidf'][j][1],acc_metrics['word2vec_tfidf'][j][2],acc_metrics['word2vec_tfidf'][j][3], median(acc_metrics['word2vec_tfidf'][j]),average(acc_metrics['word2vec_tfidf'][j])\
        ,f1_metrics['word2vec_tfidf'][j][0],f1_metrics['word2vec_tfidf'][j][1],f1_metrics['word2vec_tfidf'][j][2],f1_metrics['word2vec_tfidf'][j][3],median(f1_metrics['word2vec_tfidf'][j]),average(f1_metrics['word2vec_tfidf'][j])))

print ("####### #################################################  #################################################")

print ("""
########################################################################
base 

        time                                    
####### #################################
# NOF # #   DT  #   NB  #   RF  #  MLP  #
####### #################################""")
for j in range (len(num_features)):
    print ("""#{:^5}# # {:<5.3} # {:<5.3} # {:<5.3} # {:<5.3} #"""\
    .format(NOF[j]\
        ,str(time_metrics['base'][j][0])[:5],str(time_metrics['base'][j][1])[:5],str(time_metrics['base'][j][2])[:5],str(time_metrics['base'][j][3])[:5]))

print ("####### #################################")

print ("""
########################################################################
word2vec 

        time                                    
####### #################################
# NOF # #   DT  #   NB  #   RF  #  MLP  #
####### #################################""")
for j in range (len(num_features)):
    print ("""#{:^5}# # {:<5.3} # {:<5.3} # {:<5.3} # {:<5.3} #"""\
    .format(NOF[j]\
        ,str(time_metrics['word2vec'][j][0])[:5],str(time_metrics['word2vec'][j][1])[:5],str(time_metrics['word2vec'][j][2])[:5],str(time_metrics['word2vec'][j][3])[:5]))

print ("####### #################################")

print ("""
########################################################################
base_tfidf 

        time                                    
####### #################################
# NOF # #   DT  #   NB  #   RF  #  MLP  #
####### #################################""")
for j in range (len(num_features)):
    print ("""#{:^5}# # {:<5.3} # {:<5.3} # {:<5.3} # {:<5.3} #"""\
    .format(NOF[j]\
        ,str(time_metrics['base_tfidf'][j][0])[:5],str(time_metrics['base_tfidf'][j][1])[:5],str(time_metrics['base_tfidf'][j][2])[:5],str(time_metrics['base_tfidf'][j][3])[:5]))

print ("####### #################################")

print ("""
########################################################################
word2vec_tfidf 

        time                                    
####### #################################
# NOF # #   DT  #   NB  #   RF  #  MLP  #
####### #################################""")
for j in range (len(num_features)):
    print ("""#{:^5}# # {:<5} # {:<5} # {:<5} # {:<5} #"""\
    .format(NOF[j]\
        ,str(time_metrics['word2vec_tfidf'][j][0])[:5],str(time_metrics['word2vec_tfidf'][j][1])[:5],str(time_metrics['word2vec_tfidf'][j][2])[:5],str(time_metrics['word2vec_tfidf'][j][3])[:5]))

print ("####### #################################")