import os

import nltk  # for processing texts
import pandas as pd
import sklearn
from joblib import dump, load
from sklearn.feature_extraction.text import CountVectorizer  # to create Bag of words
from sklearn.metrics import accuracy_score, recall_score  # to calculate accuracy
from sklearn.model_selection import train_test_split  # for splitting data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder  # to convert classes to number

from .Preprocessing import clean_text
from .settings import BASE_DIR

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def knn(test_text):
    this_dir = os.path.join(BASE_DIR, 'ArabicEmotionDetector')
    if os.path.isfile(os.path.join(this_dir, 'KNNmodel.joblib')) and os.path.isfile(os.path.join(this_dir, 'KNNcount.joblib')) and os.path.isfile(os.path.join(this_dir, 'Knnencoder.joblib')):
        loaded_model = load(os.path.join(this_dir, 'KNNmodel.joblib'))
        loaded_count_vector = load(os.path.join(this_dir, 'KNNcount.joblib'))
        loaded_encoder = load(os.path.join(this_dir, 'Knnencoder.joblib'))
        test_vector = loaded_count_vector.transform(test_text)
        test_vector = test_vector.toarray()
        # encodeing predict class
        text_predict_class = loaded_encoder.inverse_transform(loaded_model.predict(test_vector))
        return text_predict_class[0]
    else:
        # douwnload data
        data = pd.read_csv(os.path.join(this_dir, 'data.csv'))
        data['text'] = data['text'].apply(clean_text)
        # create bag of words
        max_features = 20000
        count_vector = CountVectorizer(max_features=max_features)
        X = count_vector.fit_transform(data['text']).toarray()
        d = pd.DataFrame(X, columns=count_vector.get_feature_names())
        dump(count_vector, os.path.join(this_dir, 'KNNcount.joblib'))
        # convert classes to number
        encoder = LabelEncoder()
        y = encoder.fit_transform(data['Klass'])
        dump(encoder, os.path.join(this_dir, 'Knnencoder.joblib'))
        # Split data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # *********************************Classification********************************************
        # Define KNN
        model = KNeighborsClassifier(n_neighbors=5)
        # train model
        model.fit(X_train, y_train)
        # Predicting the Test set results
        # y_pred = model.predict(X_test)
        y_pred = sklearn.model_selection.cross_val_predict(model, X_test, y_test, cv=10)
        print('\nK Nearest Neighbors (NN = 3)')
        print('Accuracy Score: ', accuracy_score(y_test, y_pred) * 100, '%', sep='')
        print('rappel: ', recall_score(y_test, y_pred, average='macro', zero_division=1) * 100, '%', sep='')
        # Saving model
        dump(model, os.path.join(this_dir, 'KNNmodel.joblib'))
        # *********************************Test with new review********************************************
        # convert to number
        # test_text = clean_text(test_txt)
        test_vector = count_vector.transform(test_text)
        test_vector = test_vector.toarray()
        # encodeing predict class
        text_predict_class = encoder.inverse_transform(model.predict(test_vector))
        return text_predict_class[0]
