import re  # for pre-processing text
import string  # for pre-processing text

import stopwords

arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''  # define arabic punctuations
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations


def clean_text(text):
    """
        DESCRIPTION:
        This function to clean text
        INPUT:
        text: string
        OUTPUT:
        text: string after clean it
    """
    # remove english letters
    text = re.sub("[a-zA-Z]", " ", text)
    # remove \n from text
    text = re.sub('\n', ' ', text)
    # remove number
    text = re.sub(r'\d+', '', text)
    # remove links
    text = re.sub(r'http\S+', '', text)
    # remove punctuation
    text = text.translate(str.maketrans('', '', punctuations_list))
    # remove stop words
    text = ' '.join([word for word in text.split() if word not in stopwords.get_stopwords("arabic")])
    # remove extra space
    text = re.sub(' +', ' ', text)
    # remove whitespaces
    text = text.strip()

    return text
