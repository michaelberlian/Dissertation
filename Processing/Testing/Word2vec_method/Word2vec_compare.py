from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import time

# testing the methods of word2vec to determine which is better

num_features = [5000,10000,15000,20000]

# model preparation
DT = DecisionTreeClassifier()
NB = GaussianNB()
RF = RandomForestClassifier()
MLP = MLPClassifier()

models = [DT,NB,RF,MLP]
# models = [DT]
models_string = ["DT","NB","RF","MLP"]
counter = 0 
for model in models:
    print ("""
#############################################


{}
        {:<17}  {:<17}
####### #################  #################
# NOF # #   v1  #  v2   #  #  v1   #  v2   #
####### #################  #################""".format(models_string[counter],"accuracy","f1"))
    for num_feature in num_features:
        
        time_start = time.time()
        # data loading
        df_1 = pd.read_csv("../../../Data/Testing/Word2vec/BOW_word2vec_v1_{}.csv".format(num_feature))

        #data preparation and model fitting
        label_1 = df_1['category_label'].tolist()
        X_1 = df_1.drop(columns=['category_label'])
        X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, label_1, test_size=0.15, random_state=1000000)
        model_1 = model.fit(X_train_1,y_train_1)

        # predicting and metrics calculation
        y_pred_1 = model_1.predict(X_test_1)
        # print("time : {}".format(time.time()-time_start))
        accuracy_1 = accuracy_score(y_test_1,y_pred_1)
        f1_1 = f1_score(y_test_1,y_pred_1,average="macro")

################################################################################################################################################

        time_start = time.time()
        # data loading
        df_2 = pd.read_csv("../../../Data/Testing/Word2vec/BOW_word2vec_v2_{}.csv".format(num_feature))

        #data preparation and model fitting
        label_2 = df_2['category_label'].tolist()
        X_2 = df_2.drop(columns=['category_label'])
        X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, label_2, test_size=0.15, random_state=1000000)
        model_2 = model.fit(X_train_2,y_train_2)

        # predicting and metrics calculation
        y_pred_2 = model_2.predict(X_test_2)
        # print("time : {}".format(time.time()-time_start))
        accuracy_2 = accuracy_score(y_test_2,y_pred_2)
        f1_2 = f1_score(y_test_2,y_pred_2,average="macro")
        
        print ("""#{:^5}# # {:<5.3f} # {:<5.3f} #  # {:<5.3f} # {:<5.3f} #""".format(str(int(num_feature/1000))+'K',accuracy_1,accuracy_2,f1_1,f1_2))
    print ("####### #################  #################")
    counter += 1 