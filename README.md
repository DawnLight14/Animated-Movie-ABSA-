# Implementation of BERT and LSTM in Aspect-Based Sentiment Analysis (ABSA) on Animated Movie Review

BERT–LSTM Integrated Pipeline with Streamlit Deployment

An end-to-end Aspect-Based Sentiment Analysis (ABSA) system applied to IMDb user reviews of Pixar and DreamWorks animated films (2022–2024).

The pipeline integrates fine-tuned BERT for aspect extraction and a dual-input LSTM for sentiment classification, followed by analytical insight generation and interactive visualization.

---

## 🚀 Key Results

**Aspect Term Extraction (BERT)**
- Accuracy: 99.25%
- F1-Score: 84.74%

**Sentiment Classification (Dual-Input LSTM)**
- Accuracy: 82.08%
- F1-Macro: 82.06%

Dataset scale:
- 6,453 reviews
- 69,876 segmented sentences
- 4,075 labeled sentence–aspect pairs

---

## 🧠 What This System Does

Given a movie review, the system:

1. Extracts aspect terms (e.g., character, story, animation)
2. Classifies sentiment polarity (positive / negative) for each aspect
3. Aggregates aspect frequency
4. Visualizes sentiment distribution and dominant trends

---

## 📊 Key Insights from Analysis

Top 10 Most Discussed Aspects:

1. Character (6141)
2. Story (4980)
3. Animation (3164)
4. Plot (1588)
5. Scene (1264)
6. Voice (1067)
7. Message (965)
8. Theme (958)
9. Villain (938)
10. Humor (671)

Key Findings:
- Animation received predominantly positive feedback.
- Character and Story were widely praised.
- Plot and Villain showed relatively higher negative sentiment.

These results highlight narrative structure and visual quality as primary drivers of audience perception.

---

## 🔗 Live Demo

Streamlit Application:
https://animated-movie-absa.streamlit.app

Features:
- Input custom review text
- Automatic aspect extraction
- Aspect-level sentiment classification
- Sentiment distribution visualization
- Aspect frequency charts

---

## 🏗 System Architecture

Raw Review  
→ BERT (Aspect Extraction)  
→ Sentence–Aspect Pairing  
→ LSTM (Sentiment Classification)  
→ Aggregation & Visualization  

The architecture is modular and supports end-to-end automated analysis.

---

## 🛠 Tech Stack

Programming:
- Python
- Pandas
- NumPy
- Selenium

Machine Learning:
- HuggingFace Transformers
- BERT (bert-base-uncased)
- TensorFlow / Keras
- Dual-Input LSTM
- Scikit-learn

Deployment:
- Streamlit
- Matplotlib / Seaborn

---

## 📂 Project Structure

```
app.py
absa_pipeline.py
modules/
models/
training/
requirements.txt
```

For detailed model development and experiment documentation, see:

📁 `training/README.md`

---

## 👤 Author

Nicholas Ryan Jonathan  
Bachelor of Informatics – Data Science  
LinkedIn: https://linkedin.com/in/nichoryjo  
GitHub: https://github.com/DawnLight14
