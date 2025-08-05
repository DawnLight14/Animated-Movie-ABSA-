# modules/sentiment_classifier.py
import numpy as np

def classify_sentiment_lstm(sentence_input, aspect_input, model):
    preds = model.predict([sentence_input, aspect_input])
    pred_labels = (preds >= 0.5).astype(int).flatten()
    label_map = {0: "Negative", 1: "Positive"}  # disesuaikan dengan label training
    return [label_map[i] for i in pred_labels]
