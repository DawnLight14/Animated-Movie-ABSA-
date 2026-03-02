# Film-Level Aspect & Sentiment Analysis

This directory contains analytical scripts and notebooks used to generate aggregated insights from the integrated ABSA pipeline.

Unlike the `training/` folder (model development), this layer focuses on post-model statistical aggregation and interpretation.

---

# 🎯 Objective

To identify dominant aspects and sentiment trends across animated movie reviews from Pixar and DreamWorks (2022–2024).

The analysis aims to answer:

- What aspects are most frequently discussed?
- Which aspects receive predominantly positive or negative sentiment?
- What narrative components drive audience perception?

---

# 📊 Analytical Components

## 1️⃣ Aspect Frequency Analysis

Steps:
- Collect all extracted aspects from the ABSA pipeline
- Normalize tokens
- Aggregate total occurrences
- Rank aspects by frequency

Output:
- Top-N most discussed aspects
- Bar chart visualization
- CSV summary

---

## 2️⃣ Sentiment Distribution per Aspect

Steps:
- Group sentence–aspect pairs
- Calculate positive vs negative counts
- Compute percentage distribution
- Generate comparative visualizations

Output:
- Sentiment distribution tables
- Bar charts
- Insight summary

---

## 3️⃣ Film-Level Aggregation

Optional (if applied):
- Group analysis by movie
- Compare dominant aspects per film
- Identify variation in sentiment polarity

---

# 📈 Key Findings

From 6,453 reviews:

- Character and Story are the most dominant aspects.
- Animation consistently receives strong positive sentiment.
- Plot and Villain show comparatively higher negative polarity.
- Emotional engagement and narrative coherence strongly influence review tone.

These findings align with qualitative patterns observed in IMDb ratings.

---

# 🔬 Methodological Notes

- Aggregation performed after full ABSA inference
- Duplicate aspect normalization applied
- Sentiment classified using trained LSTM model
- Statistical summaries computed using Pandas

---

# 📌 Purpose of This Layer

This directory demonstrates:

- Data interpretation capability
- Statistical aggregation skills
- Insight generation from NLP output
- Translation of model output into business-level findings

It bridges the gap between machine learning output and decision-support insight.
