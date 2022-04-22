from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import joblib
import numpy as np
##############################################################################################################################

# BOW-methods
bows = ["base","base_tfidf","word2vec","word2vec_tfidf"]
start_time = time.time()
for bow in bows:

    print ("####################################")
    # start_time = time.time()
    # data loading and preparation 
    df = pd.read_csv("../Data/Processed/BOW_{}.csv".format(bow))

    label = df['category_label'].tolist()
    X = df.drop(columns=['category_label'])
    X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.15, random_state=1000000)

    # smoothing = [0.000000001,0.00000001,0.0000001,0.000001,0.0001,0.001,0.01,0.1,1]
    # hyper-parameter candidates
    smoothing = np.logspace(0,-9,50)

    # build param_grid dictionary
    param_grid = {'var_smoothing': smoothing,}

    # initiate model and grid search
    DT = GaussianNB()
    model = GridSearchCV(DT, param_grid)
    
    # fit the grid search
    tuned = model.fit(X_train, y_train)
    print(tuned.best_params_)

    # save the grid search result
    joblib.dump(model,"../Model/Tuned/NB_{}.joblib".format(bow))

print ("time: {}".format(time.time()-start_time))