# pipeline.py
import os
import torch
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from transformers import AutoTokenizer, AutoModelForTokenClassification
from modules.preprocessing_umum import preprocessing_umum, segment_sentences
from modules.aspect_extraction import extract_aspects
from modules.lstm_preprocessing import preprocess_pairs, tokenize_and_pad_lstm
from modules.sentiment_classifier import classify_sentiment_lstm


def run_absa_pipeline(review_text):
    # Preprocessing
    cleaned_text = preprocessing_umum(review_text)
    sentences = segment_sentences(cleaned_text)

    # Load BERT model & tokenizer
    bert_path = os.path.join("models", "bert_ate_finetuned")
    tokenizer_bert = AutoTokenizer.from_pretrained(bert_path)
    model_bert = AutoModelForTokenClassification.from_pretrained(bert_path)
    device = 0 if torch.cuda.is_available() else -1

    # Aspect Extraction
    extracted_aspects = extract_aspects(sentences, tokenizer_bert, model_bert, device=device)

    # Load LSTM tokenizers
    with open(os.path.join("models", "tokenizer_sentence.pkl"), "rb") as f:
        tokenizer_sentence = pickle.load(f)
    with open(os.path.join("models", "tokenizer_aspect.pkl"), "rb") as f:
        tokenizer_aspect = pickle.load(f)

    # Prepare LSTM inputs
    sentence_aspect_pairs = preprocess_pairs(sentences, extracted_aspects)

    X_sentence, X_aspect = tokenize_and_pad_lstm(
    sentence_aspect_pairs, 
    tokenizer_sentence, 
    tokenizer_aspect, 
    maxlen_sentence=500,
    maxlen_aspect=4
    )

    # Load and compile LSTM model
    # print("🚀 Loading LSTM model...")
    lstm_model_path = os.path.join("models", "lstm_model.h5")
    lstm_model = load_model(lstm_model_path)
    # print("✅ Loaded model:", type(lstm_model))
    lstm_model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=0.0005),
              metrics=['accuracy'])

    # Sentiment classification
    predictions = classify_sentiment_lstm(X_sentence, X_aspect, lstm_model)

    # Return list of (sentence, aspect, sentiment)
    results = [
        {"sentence": s, "aspect": a, "sentiment": p} 
        for (s, a), p in zip(sentence_aspect_pairs, predictions)
    ]
    return results
