import os
import re

import pandas as pd
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from .Preprocessing import clean_text

from ArabicEmotionDetector.settings import JOBLIB_DIR, DATA_DIR


def pre_processing():
    train = pd.read_csv(os.path.join(DATA_DIR, 'balanced_2classes_train.csv'), encoding='utf-8')
    val = pd.read_csv(os.path.join(DATA_DIR, 'balanced_2classes_val.csv'), encoding='utf-8')
    test = pd.read_csv(os.path.join(DATA_DIR, 'balanced_2classes_test.csv'), encoding='utf-8')
    stops = pd.read_excel(os.path.join(DATA_DIR, 'ar_stops.xlsx'), encoding='utf-8')
    frames = [train, val, test]

    data = pd.concat(frames)

    data['text'] = data['text'].apply(lambda x: re.sub(r"[0-9]", " ", x))
    stop_words = list(stops['“'])
    review = data['text']
    review = review.apply(lambda x: " ".join(x))
    train = data[:train.shape[0]]
    val = data[train.shape[0]: (val.shape[0] + train.shape[0])]
    test = data[(val.shape[0] + train.shape[0]):]

    X_train = train['text']
    y_train = train['label']

    if not os.path.isfile(os.path.join(JOBLIB_DIR, 'count_vector.joblib')):
        train_df_vectorized = TfidfVectorizer(min_df=2, ngram_range=(1, 3))
        X = train_df_vectorized.fit_transform(X_train)
        dump(train_df_vectorized.fit(X_train), os.path.join(JOBLIB_DIR, 'count_vector.joblib'))
    else:
        train_df_vectorized = load(os.path.join(JOBLIB_DIR, 'count_vector.joblib'))
        X = train_df_vectorized.fit_transform(X_train)

    return [X, y_train]


def predict(model, count_vector, text_to_predict):
    prediction = count_vector.transform(text_to_predict)
    p = model.predict(prediction)[0]
    if p == 0:
        return "Negative"
    elif p == 1:
        return "Positive"
    else:
        return "Negative"


def train_naive_bayes():
    data = pre_processing()

    clfrNB = MultinomialNB(alpha=0.1)
    clfrNB.fit(data[0], data[1])

    cv = cross_val_score(clfrNB, data[0], data[1], cv=10)

    dump(clfrNB, os.path.join(JOBLIB_DIR, 'NBmodel.joblib'))


def predict_naive_bayes(text_to_predict):
    if not os.path.isfile(os.path.join(JOBLIB_DIR, 'NBmodel.joblib')) or not os.path.isfile(os.path.join(JOBLIB_DIR, 'count_vector.joblib')):
        train_naive_bayes()

    loaded_model = load(os.path.join(JOBLIB_DIR, 'NBmodel.joblib'))
    loaded_count_vector = load(os.path.join(JOBLIB_DIR, 'count_vector.joblib'))

    return predict(loaded_model, loaded_count_vector, text_to_predict)


def train_knn():
    data = pre_processing()

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(data[0], data[1])
    dump(model, os.path.join(JOBLIB_DIR, 'KNNmodel.joblib'))

    cv2 = cross_val_score(model, data[0], data[1], cv=10)


def predict_knn(text_to_predict):
    if not os.path.isfile(os.path.join(JOBLIB_DIR, 'KNNmodel.joblib')) or not os.path.isfile(
            os.path.join(JOBLIB_DIR, 'count_vector.joblib')):
        train_knn()

    loaded_model = load(os.path.join(JOBLIB_DIR, 'KNNmodel.joblib'))
    loaded_count_vector = load(os.path.join(JOBLIB_DIR, 'count_vector.joblib'))

    return predict(loaded_model, loaded_count_vector, text_to_predict)


def train_svm():
    data = pre_processing()

    svm = SVC(kernel='linear')
    svm.fit(data[0], data[1])

    cv2 = cross_val_score(svm, data[0], data[1], cv=10)

    dump(svm, os.path.join(JOBLIB_DIR, 'SVMmodel.joblib'))


def predict_svm(text_to_predict):
    if not os.path.isfile(os.path.join(JOBLIB_DIR, 'SVMmodel.joblib')) or not os.path.isfile(
            os.path.join(JOBLIB_DIR, 'count_vector.joblib')):
        train_svm()

    loaded_model = load(os.path.join(JOBLIB_DIR, 'SVMmodel.joblib'))
    loaded_count_vector = load(os.path.join(JOBLIB_DIR, 'count_vector.joblib'))

    return predict(loaded_model, loaded_count_vector, text_to_predict)


def lexicon_dictionary(text_to_predict):
    # Lexicon dictionary
    text_to_predict = clean_text(text_to_predict)
    lexicon = dict()
    # Read in the lexicon.
    df = pd.read_csv(os.path.join(DATA_DIR, 'SemEval2016-Arabic.csv'), delimiter=',')
    for row in df.iterrows():
        lexicon[row[1].word] = float(row[1].score)
    # Use lexicon to score tweets
    score = 0
    for word in text_to_predict.split():
        if word in lexicon:
            score = score + lexicon[word]

    if score > 0:
        sentiment = 'Positive'
    else:
        sentiment = 'Negative'

    return sentiment
