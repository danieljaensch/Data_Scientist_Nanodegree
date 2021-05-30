# --- import libraries ---
import pandas as pd
import numpy as np
import sys
import re
import pickle
import nltk

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection  import GridSearchCV
from sklearn import multioutput
# ---------------------

# disable warnings
import warnings
warnings.filterwarnings("ignore")

# download nltk packages
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def load_data(database_filepath):
    '''
    Load datasets from local SQLite database

    @param: database_filepath - string - filename for SQLite database containing cleaned message data.

    @returns:
    X - dataframe - Dataframe containing features dataset.
    y - dataframe - Dataframe containing labels dataset.
    col_names - List of strings - List containing category names.
    '''
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('InsertTableName', engine)

    # drop nan values
    df.dropna(axis=0, how = 'any', inplace = True)

    X = df['message']
    y = df.iloc[:,4:].astype(int)
    col_names = y.columns.values

    return X, y, col_names



def tokenize(text):
    '''
    Receives as input raw text which afterwards normalized, stop words removed,
    stemmed and lemmatized.

    @param  - text - input raw text
    @return - clean_tokens - tokenized text as result
    '''

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []

    for token in tokens:
        clean_token = lemmatizer.lemmatize( token ).lower().strip()
        clean_tokens.append(clean_token)
    # --- for ---

    return clean_tokens



def build_model():
    '''
    Build a machine learning pipeline

    @param - None
    @returns - cv - gridsearch-cv object - Gridsearchcv object that transforms the data, creates the
                                            model object and finds the optimal model parameters.
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier( AdaBoostClassifier() ))
    ])

    # setting model parameters
    parameters = {'vect__min_df': [5],
                  'tfidf__use_idf':[True],
                  'clf__estimator__learning_rate': [0.5, 1],
                  'clf__estimator__n_estimators':[10, 25]
                 }

    cv = GridSearchCV( pipeline, param_grid=parameters )

    return cv



def get_evaluation(y_test, y_pred, category_names):
    '''
    Evaluate model performance based on accuracy, precision, recall, F1 score

    @param - y_test - array - Actual labels.
    @param - y_pred - array - Predicted labels.
    @param - category_names - list of strings - List containing names for each of the predicted fields.

    @returns - df - dataframe - Dataframe containing the accuracy, precision, recall
                                and f1 score for a given set of actual and predicted labels.
    '''

    metrics = []
    for i in range(len(category_names)):
        accuracy = accuracy_score(y_test[:,i], y_pred[:,i])
        precision = precision_score(y_test[:,i], y_pred[:,i], average='micro')
        recall = recall_score(y_test[:,i], y_pred[:,i], average='micro')
        f1 = f1_score(y_test[:,i], y_pred[:,i], average='micro')
        # add to metrics
        metrics.append([accuracy, precision, recall, f1])
    # ---- for ----

    df = pd.DataFrame(data = np.array(metrics),
                      index=category_names,
                      columns=['Accuracy', 'Precision', 'Recall', 'F1 score'])

    return df



def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model performance based on accuracy, precision, recall, F1 score

    @param - actual - array - Array containing actual labels.
    @param - predicted - array - Array containing predicted labels.
    @param - category_names - list of strings - List containing names for each of the predicted fields.

    @returns - df - dataframe - Dataframe containing the accuracy, precision, recall
                                and f1 score for a given set of actual and predicted labels.
    '''

    Y_pred = model.predict(X_test)
    # display / print result of model evaluation
    print( get_evaluation(np.array(Y_test), Y_pred, category_names) )



def save_model(model, model_filepath):
    '''
    Saving models best-estimator using pickle

    @param - model - the models best-estimator
    @param - model_filepath - string - the file path to save the model
    '''
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


# ------------------------------------------------------------------------------
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    main()
