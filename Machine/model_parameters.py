import joblib
# to print out the best parameters from each tuned models
bows = ["base","base_tfidf","word2vec","word2vec_tfidf"]

for bow in bows:
    DT_model = joblib.load('../Model/Tuned/DT_{}.joblib'.format(bow))
    NB_model = joblib.load('../Model/Tuned/NB_{}.joblib'.format(bow))
    RF_model = joblib.load('../Model/Tuned/RF_{}.joblib'.format(bow))
    MLP_model = joblib.load('../Model/Tuned/MLP_{}.joblib'.format(bow))
    print ("DT_{} : {}".format(bow,DT_model.best_params_))
    print ("NB_{} : {}".format(bow,NB_model.best_params_))
    print ("RF_{} : {}".format(bow,RF_model.best_params_))
    print ("MLP_{} : {}".format(bow,MLP_model.best_params_))