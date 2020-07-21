import os

import pandas as pd
import sklearn
from joblib import dump, load
from sklearn.feature_extraction.text import CountVectorizer  # to create Bag of words
from sklearn.metrics import accuracy_score, recall_score  # to calculate accuracy
from sklearn.model_selection import train_test_split  # for splitting data
from sklearn.preprocessing import LabelEncoder  # to convert classes to number
from sklearn.svm import SVC

from .Preprocessing import clean_text


def svm(test_text):
    if os.path.isfile('SVMmodel.joblib') and os.path.isfile('SVMencoder.joblib') and os.path.isfile('SVMcount.joblib'):
        loadedmodel = load('SVMmodel.joblib')
        loaded_encoder = load('SVMencoder.joblib')
        loaded_count = load('SVMcount.joblib')
        max_features = 200000
        test_vector = loaded_count.transform(test_text)
        test_vector = test_vector.toarray()
        # encodeing predict class
        text_predict_class = loaded_encoder.inverse_transform(loadedmodel.predict(test_vector))
        return text_predict_class[0]
    else:
        # download data
        data = pd.read_csv('data.csv')
        data['text'] = data['text'].apply(clean_text)
        # create bag of words
        max_features = 200000
        count_vector = CountVectorizer(max_features=max_features)
        X = count_vector.fit_transform(data['text']).toarray()
        d = pd.DataFrame(X, columns=count_vector.get_feature_names())
        dump(count_vector, 'SVMcount.joblib')
        # convert classes to number
        encoder = LabelEncoder()
        y = encoder.fit_transform(data['Klass'])
        dump(encoder, 'SVMencoder.joblib')
        # Split data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # *********************************Classification********************************************
        # Define SVM
        model = SVC(kernel='rbf')
        # train model
        model.fit(X_train, y_train)
        # Predicting the Test set results
        # y_pred = model.predict(X_test)
        y_pred = sklearn.model_selection.cross_val_predict(model, X_test, y_test, cv=10)
        print('\nSupport Vector Machine')
        print('Accuracy Score: ', accuracy_score(y_test, y_pred) * 100, '%', sep='')
        print('rappel: ', recall_score(y_test, y_pred, average='macro', zero_division=1) * 100, '%', sep='')
        dump(model, 'SVMmodel.joblib')
        # *********************************Test with new review********************************************
        # convert to number
        test_vector = count_vector.transform(test_text)
        test_vector = test_vector.toarray()
        # encodeing predict class
        text_predict_class = encoder.inverse_transform(model.predict(test_vector))
        return text_predict_class[0]
