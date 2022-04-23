The programs were run and tested on Python 3.8.9, <br>
the libraries that were used are : <br>
- Selenium version 3.141.0 (https://selenium-python.readthedocs.io)
- Numpy version 1.21.3 (https://numpy.org)
- Pandas version 1.3.4 (https://pandas.pydata.org)
- Gensim version 4.1.2 (https://radimrehurek.com/gensim/)
- Natural Language ToolKit version 3.6.5 (https://radimrehurek.com/gensim/)
- Scikit Learn version 1.0 (https://scikit-learn.org)
- Joblib version 1.1.0 (https://joblib.readthedocs.io/en/latest/)

chromedriver is needed to run selenium scraping (https://sites.google.com/chromium.org/driver/)

Folder :<br>
- Data : contains xlsx, csv, and pickles.
    - Raw : contains dataset from scraping and kaggle
    - Cleaned : Dataset that had been through cleaning and sampled
    - Processed : Contain Cleaned and final dataset to be fitted to model 
        - Eval : processed dataset to be tested against model (evaluation)
    - Pickles : contains list and dictionary neccessary to perform algorithms
    - Testing : contains csvs needed to perform testing 
- Model : contains .model and .joblib
    - Testing/Word2vec : contains models to compare word2vec models' parameters and pre-trained word2vec models
    - Tuned : contains models that had been tuned using GridSearchCV by program in machine folder
    - Word2vec : contains final model of Word2vec algorithm
- Scraping : python file to scrape from BBC(https://www.bbc.co.uk/) and CNN(https://edition.cnn.com/) news website, results saved at Data/Raw
- Processing : python file to clean and pre-process the data
    - Testing : testing multiple methods and parameters on pre-processing steps
    - Cleaning.py : python program to clean and remove redundancy from raw dataset, results saved at Data/Cleaned, Data/Processed and Data/Processed/Eval
    - Word2vec.py : python program to create word2vec model, model saved at Model/Word2vec
    - Word2vec_apply.py : applying algorithm using built word2vec model to cleaned data, results saved at Data/Processed, Data/Processed/Eval, and Data/Pickles
    - Bag-Of-Word.py : creating Bag-of-Words of multiple methods CSVs from datasets, results saved at Data/Processed, Data/Processed/Eval, and Data/Pickles
- Machine : python file to fit, tune, and evaluate models
    - pre-tuning.py : early fit and evaluation without model tuning
    - DT_tuning.py : Fitting and tuning for Decision Tree
    - NB_tuning.py : Fitting and tuning for Naive Bayes
    - RF_tuning.py : Fitting and tuning for Random Forest 
    - MLP_tuning.py : Fitting and tuning for Multi-Layer Perceptron
    - model_eval.py : final evaluation for saved tuned model
- Result : contain screenshots and xlsx of results and evaluation

before running python files download and UNZIP Data.zip and Model.zip to get Data and Model Folders. <br>

data can be found and downloaded here : 
- https://drive.google.com/drive/folders/1pGaORhuollYKH3rDDAXxBOjhbxMzuLG4?usp=sharing (Data.zip and Model.zip)

Repository to the source code can be found here : 
- https://drive.google.com/drive/folders/1H7BtlzoJOagq4y-AAXN0UNkH9yOYYWF8?usp=sharing (source code )or
- https://github.com/michaelberlian/Dissertation-Article-Category-Prediction (source code only) or
- https://projects.cs.nott.ac.uk/psymb17/Dissertation-Article-Category-Prediction (source code only)

