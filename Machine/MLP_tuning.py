from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
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

    # hidden_layer = [(100),(50)]
    # activation = ["relu"]
    # max_iter = [200]
    # hyper-parameter candidates
    hidden_layer = [(100),(150),(100,150),(100,100)]
    activation = ["logistic", "relu"]
    max_iter = [100,200,300]

    # build param_grid dictionary
    param_grid = {'hidden_layer_sizes': hidden_layer, 
                'activation': activation,
                'max_iter': max_iter}

    # initiate model and grid search
    MLP = MLPClassifier()
    model = GridSearchCV(MLP, param_grid)

    # fit the grid search
    tuned = model.fit(X_train, y_train)
    print(tuned.best_params_)

    # save the grid search result
    joblib.dump(model,"../Model/Tuned/MLP_{}.joblib".format(bow))

print ("time: {}".format(time.time()-start_time))