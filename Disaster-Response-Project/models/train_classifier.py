# import libraries
import sys
import numpy as np
import pandas as pd
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
import pickle



def load_data(database_filepath):
    '''Reads in the data from the specified database filepath
    and return the dataframe along with categories
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    table_name = 'disaster'
    df = pd.read_sql_table(table_name,engine)
    X = df['message']
    y = df[['related', 'request', 'offer',
           'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
           'security', 'military', 'child_alone', 'water', 'food', 'shelter',
           'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
           'infrastructure_related', 'transport', 'buildings', 'electricity',
           'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
           'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
           'other_weather', 'direct_report']]
    categories = y.columns
    
    return X,y,categories

def tokenize(text):
    ''' Function takes in text as a parameter
    normalizes the text by converting it to lowercase and removing the punctuation symbols.
    Splits the text to individual tokens and removes the stopwords from the text.
    Lemmatize the tokens and return them.
    '''
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9]',' ',text)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    tokens = [WordNetLemmatizer().lemmatize(token).strip() for token in tokens]
    return tokens


def build_model():
    '''Create a Pipeline containing two estimators CountVectorizer and TfidfTransformer
    and a predictor Random Forest Classifier
    '''
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])
    #Reducing Parameters because of large pickle file issue
    parameters = { 'clf__estimator__min_samples_split': [10],
                   'clf__estimator__n_estimators': [2]
                }

    model = GridSearchCV(pipeline,param_grid = parameters,verbose = 3,n_jobs = 4,cv = 3)
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    '''Evaluate the Model and prints the model report'''
    Y_pred = model.predict(X_test)
    model_report = classification_report(Y_test,Y_pred,target_names=category_names)
    print(model_report)


def save_model(model, model_filepath):
    '''Saves the model parameters in the path mentioned'''
    with open(model_filepath,'wb') as f:
        pickle.dump(model,f)
        


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


if __name__ == '__main__':
    main()