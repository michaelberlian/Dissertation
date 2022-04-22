from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import joblib
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

    # hyper-parameter candidates
    estimator = [100,150,200,250]
    criterion = ["gini", "entropy"]
    depth = [100,200,300,None]

    # build param_grid dictionary
    param_grid = {'n_estimators': estimator, 
                'criterion': criterion,
                'max_depth': depth}

    # initiate model and grid search
    RF = RandomForestClassifier()
    model = GridSearchCV(RF, param_grid)

    # fit the grid search
    tuned = model.fit(X_train, y_train)
    print(tuned.best_params_)
    
    # save the grid search result
    joblib.dump(model,"../Model/Tuned/RF_{}.joblib".format(bow))
    
print ("time: {}".format(time.time()-start_time))