# Implementation of BERT and LSTM in Aspect-Based Sentiment Analysis (ABSA) on Animated Movie Review

BERT–LSTM Integrated Pipeline with Analytical Insight Generation

An end-to-end Aspect-Based Sentiment Analysis (ABSA) system applied to IMDb user reviews of Pixar and DreamWorks animated films (2022–2024).

The project integrates:
- Fine-tuned BERT for aspect extraction
- Dual-input LSTM for aspect-level sentiment classification
- Film-level analytical aggregation
- Interactive Streamlit deployment

---

## 🚀 Key Results

Aspect Term Extraction (BERT)
- Accuracy: 99.25%
- F1-Score: 84.74%

Aspect Sentiment Classification (Dual-Input LSTM)
- Accuracy: 82.08%
- F1-Macro: 82.06%

Dataset Scale:
- 6,453 reviews
- 69,876 segmented sentences
- 4,075 labeled sentence–aspect pairs

---

## 🧠 System Capabilities

Given a movie review, the system:

1. Extracts aspect terms (e.g., character, story, animation)
2. Classifies sentiment polarity (positive / negative) for each aspect
3. Aggregates aspect frequency across films
4. Computes sentiment distribution per aspect
5. Generates visual summaries (bar charts)

---

## 📊 Film-Level Analytical Insights

The system performs post-model aggregation to identify dominant narrative elements and audience perception trends.

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

Key Observations:
- Animation received predominantly positive sentiment.
- Character and Story were the most influential aspects.
- Plot and Villain showed relatively higher negative polarity.
- Narrative depth and visual execution strongly influenced audience ratings.

Detailed analytical notebooks are available in:

📁 `analysis/`

---

## 🏗 Project Structure

```
app.py
absa_pipeline.py

modules/
models/

training/      → model development & experiments
analysis/      → film-level aggregation & insight generation

requirements.txt
README.md
```

---

## 🌐 Live Demo

Streamlit Application:
https://animated-movie-absa.streamlit.app

Features:
- Custom review input
- Automatic aspect extraction
- Aspect-level sentiment classification
- Sentiment distribution visualization
- Dominant aspect identification

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

Visualization & Deployment:
- Matplotlib
- Seaborn
- Streamlit

---

## 📌 Repository Layers

training/ → hyperparameter tuning, experiments, model evaluation  
analysis/ → aggregated film-level insight & statistical summaries  
app.py → deployed ABSA pipeline  

This separation ensures modular development, reproducibility, and clean research-to-deployment workflow.

---

## 👤 Author

Nicholas Ryan Jonathan  
Bachelor of Informatics – Data Science  
LinkedIn: https://linkedin.com/in/nichoryjo  
GitHub: https://github.com/DawnLight14
