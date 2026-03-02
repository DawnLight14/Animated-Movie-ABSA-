# Dataset Samples

This folder provides structured sample datasets used for training and evaluating the ABSA pipeline.

⚠ The full IMDb scraped dataset is not publicly distributed due to platform terms of service.  
Only representative samples are included for demonstration and reproducibility purposes.

---

## 📦 Original Dataset Scale

- 6,453 raw IMDb reviews
- 69,876 segmented sentences
- 6,987 BIO-tagged sentences (10% stratified sampling)
- 4,075 labeled sentence–aspect–sentiment pairs
- 11 animated films (Pixar & DreamWorks, 2022–2024)

---

## 📁 Files Included

### 1️⃣ movie_review_tokenized.json
Format used for BERT-based Aspect Term Extraction (ATE).

Structure:
- tokens
- labels (BIO scheme)

Purpose:
Demonstrates token-level annotation for aspect extraction.

---

### 2️⃣ Movie_Review_Processed_v4.csv
Format used for LSTM-based Aspect Sentiment Classification (ASC).

Structure:
- sentence
- aspect
- sentiment (0 = negative, 1 = positive)

Purpose:
Shows supervised learning format for sentence–aspect pairs.

---

### 3️⃣ pipeline_result.csv
Output example from the integrated ABSA pipeline.

Structure:
- sentence
- extracted aspect
- Predicted_Sentiment

Purpose:
Demonstrates end-to-end system capability.

---

## 🧠 Annotation Details

Aspect Extraction:
- BIO tagging scheme
- Token-level manual annotation

Sentiment Classification:
- Binary polarity (positive / negative)
- Neutral sentiment excluded for classification consistency

---

## 📌 Reproducibility Note

Researchers may reconstruct the full dataset by following the data collection and preprocessing methodology described in the thesis and repository documentation.

All preprocessing scripts are available in the `modules/` and `training/` directories.
