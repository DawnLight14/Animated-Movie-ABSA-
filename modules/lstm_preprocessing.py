# modules/lstm_preprocessing.py
import re
import string
import emoji
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences

lemmatizer = WordNetLemmatizer()

def preprocessing_lstm(text):
    if pd.isna(text):
        return ''
    text = text.lower()
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(f"[{string.punctuation}]", '', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)  # hapus huruf berulang lebih dari 2
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words).strip()

def preprocess_pairs(sentences, extracted_aspects):
    pairs = []
    for sentence, aspects in zip(sentences, extracted_aspects):
        for aspect in aspects:
            pairs.append((preprocessing_lstm(sentence), preprocessing_lstm(aspect)))
    return pairs

def tokenize_and_pad_lstm(pairs, tokenizer_sentence, tokenizer_aspect, maxlen_sentence=500, maxlen_aspect=4):
    sentence_seqs = tokenizer_sentence.texts_to_sequences([p[0] for p in pairs])
    aspect_seqs = tokenizer_aspect.texts_to_sequences([p[1] for p in pairs])

    sentence_padded = pad_sequences(sentence_seqs, maxlen=maxlen_sentence, padding='post', truncating='post')
    aspect_padded = pad_sequences(aspect_seqs, maxlen=maxlen_aspect, padding='post', truncating='post')

    return sentence_padded, aspect_padded
