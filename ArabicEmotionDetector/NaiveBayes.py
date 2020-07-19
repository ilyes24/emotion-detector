from sklearn.feature_extraction.text import CountVectorizer # to create Bag of words
from sklearn.model_selection import train_test_split  # for splitting data
from sklearn.naive_bayes import GaussianNB # to bulid classifier model
from sklearn.preprocessing import LabelEncoder # to convert classes to number
from sklearn.metrics import accuracy_score # to calculate accuracy
import nltk # for processing texts
import pandas as pd
from nltk.corpus import stopwords # list of stop words

from Preprocessing import clean_text

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
def NaiveBayes():
    # douwnload data
    data = pd.read_csv('data.csv')
    data = data.drop(columns=['ar:manual_confidence'])  # remove last column
    data = data.rename(columns={"Arabic_text": "text", "ar:manual_sentiment": "sentiment"})  # rename columns name
    data['text'] = data['text'].apply(clean_text)
    # create bag of words
    max_features = 1500
    count_vector = CountVectorizer(max_features=max_features)
    X = count_vector.fit_transform(data['text']).toarray()
    d = pd.DataFrame(X, columns=count_vector.get_feature_names())

    # convert classes to number
    encoder = LabelEncoder()
    y = encoder.fit_transform(data['sentiment'])

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #*********************************Classification********************************************
    # Define Gauusian Naive bayes
    model = GaussianNB()
    # train model
    model.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = model.predict(X_test)
    y_pred

    print('Test model accuracy: ', accuracy_score(y_test, y_pred))