
import re # for preprocessing text
import string # for preprocessing text
arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ''' # define arabic punctuations
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations

def clean_text(text):
  '''
  DESCRIPTION:
  This function to clean text
  INPUT:
  text: string
  OUTPUT:
  text: string after clean it
  '''
  text = re.sub("[a-zA-Z]", " ", text) # remove english letters
  text = re.sub('\n', ' ', text) # remove \n from text
  text = re.sub(r'\d+', '', text) #remove number
  text = re.sub(r'http\S+', '', text) # remove links
  text = text.translate(str.maketrans('','', punctuations_list)) # remove punctuation
  text = ' '.join([word for word in text.split() if word not in stopwords.words("arabic")]) # remove stop word
  text = re.sub(' +', ' ',text) # remove extra space
  text = text.strip() #remove whitespaces

  return text