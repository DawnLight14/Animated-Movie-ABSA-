# modules/preprocessing_umum.py
import re
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')

def preprocessing_umum(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,!?;:])([^\s])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def segment_sentences(text):
    return sent_tokenize(text)