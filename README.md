# Implementation of BERT and LSTM in Aspect-Based Sentiment Analysis (ABSA) on Animated Movie Review

An end-to-end Aspect-Based Sentiment Analysis (ABSA) system applied to IMDb user reviews of Pixar and DreamWorks animated films released between 2022–2024.

This project was developed as an undergraduate research thesis and implements a modular research-to-deployment pipeline integrating deep learning models with film-level analytical insight generation.

---

## 🎯 Project Objectives

1. Evaluate the effectiveness of a BERT–LSTM integrated pipeline for Aspect-Based Sentiment Analysis.
2. Identify dominant narrative aspects in animated movie reviews.
3. Analyze sentiment polarity trends across key film components.
4. Deploy the system as an interactive web-based application using Streamlit.

---

## 🧠 Model Architecture

The ABSA system consists of two main components:

### 1️⃣ Aspect Term Extraction (ATE)
- Model: Fine-tuned BERT (bert-base-uncased)
- Labeling Scheme: BIO (Beginning–Inside–Outside)
- Task: Token-level classification

### 2️⃣ Aspect Sentiment Classification (ASC)
- Model: Dual-Input LSTM
- Inputs: (sentence, aspect) pairs
- Output: Binary sentiment (positive / negative)

Both models were trained independently using optimal hyperparameters and then integrated into a unified ABSA pipeline.

---

## 🚀 Performance Results

### Aspect Extraction (BERT)
- Accuracy: 99.25%
- F1-Score: 84.74%

### Sentiment Classification (LSTM)
- Accuracy: 82.08%
- F1-Macro: 82.06%

These results demonstrate strong contextual extraction capability from BERT and stable sentiment classification performance from the dual-input LSTM architecture.

---

## 📊 Dataset Overview

Source: IMDb User Reviews (Web Scraping via Selenium)

- 6,453 raw reviews
- 69,876 segmented sentences
- 6,987 BIO-tagged sentences (10% stratified sampling)
- 4,075 labeled sentence–aspect–sentiment pairs
- 11 animated films (Pixar & DreamWorks, 2022–2024)

⚠ Due to IMDb terms of service, the full scraped dataset is not publicly distributed.  
Sample structured data is provided in the `data/` directory for demonstration and reproducibility purposes.

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

### Key Observations:
- Animation received predominantly positive sentiment.
- Character and Story were the most influential aspects in audience perception.
- Plot and Villain showed relatively higher negative polarity.
- Narrative depth and visual execution strongly influenced overall audience evaluation.

Detailed analytical notebooks are available in:

📁 `analysis/`

---

## 🏗 Repository Structure

```
app.py → Streamlit interface
absa_pipeline.py → Integrated ABSA pipeline

modules/ → Preprocessing & model utilities
models/ → Fine-tuned model assets

training/ → Model development & experiments
analysis/ → Film-level aggregation & insight generation
data/ → Sample structured datasets

requirements.txt
README.md
```
This layered separation ensures modularity, reproducibility, and clean research-to-deployment workflow.

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
