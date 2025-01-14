from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
import string
import re
import json

STOP_WORDS = [
    'a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am',
    'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because',
    'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can',
    'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn',
    "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for',
    'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have',
    'haven', "haven't", 'having', 'he', 'her', 'here', 'hers', 'herself', 'him',
    'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn', "isn't",
    'it', "it's", 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn',
    "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn',
    "needn't", 'nor', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or',
    'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's',
    'same', 'shan', "shan't", 'she', "she's", 'should', "should've", 'shouldn',
    "shouldn't", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the',
    'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they',
    'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've',
    'very', 'was', 'wasn', "wasn't", 'we', 'were', 'weren', "weren't", 'what',
    'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with',
    'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll",
    "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'
]

def preprocess_text(text):
    text = text.lower()
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    # Remove non-alphabetic characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Convert to lower case
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in STOP_WORDS)
    return text

def main():
    data = pd.read_csv('./text.csv')
    data['cleaned_text'] = data['text'].apply(preprocess_text)
    data = data[data['cleaned_text'] != ''].reset_index(drop=True)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data['cleaned_text'])

    tokenizer_json = tokenizer.to_json()
    with open('tokenizer.json', 'w') as f:
        f.write(tokenizer_json)

if __name__== "__main__":
    main()
