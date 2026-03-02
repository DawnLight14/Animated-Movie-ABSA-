# Model Development & Experiment Documentation

This document describes the technical methodology, experiments, and evaluation of the BERT-based Aspect Term Extraction (ATE) model and the LSTM-based Aspect Sentiment Classification (ASC) model.

---

# 1️⃣ Dataset Preparation

Source:
IMDb User Reviews (Pixar & DreamWorks films, 2022–2024)

Data Statistics:
- 6,453 raw reviews
- 69,876 segmented sentences
- 6,987 manually annotated sentences (10% stratified sampling)
- 4,075 sentence–aspect–sentiment pairs

Preprocessing Steps:

For Aspect Extraction:
- Text cleaning (URLs, spacing, punctuation normalization)
- Sentence segmentation
- Tokenization using `bert-base-uncased`
- BIO tagging (B-ASP, I-ASP, O)

For Sentiment Classification:
- Lowercasing
- Lemmatization
- Punctuation & emoji removal
- Sentence–aspect pairing

---

# 2️⃣ BERT for Aspect Term Extraction (ATE)

Model:
- bert-base-uncased (HuggingFace Transformers)
- Token classification with BIO scheme

Training Setup:
- Epochs: 3
- Optimizer: AdamW (weight decay 0.01)
- GPU training

## Hyperparameter Experiments

Best Configuration:
- Learning Rate: 2e-5
- Batch Size: 8

Performance:
- Validation Accuracy: 99.25%
- F1-Score: 84.74%

Observations:
- Strong performance in multi-aspect sentences.
- Limitations in multi-word aspect detection.
- Occasional misclassification in syntactically complex sentences.

---

# 3️⃣ LSTM for Aspect Sentiment Classification (ASC)

Architecture:
- Dual-input model
  - Separate embedding + LSTM for sentence
  - Separate embedding + LSTM for aspect
- Concatenation layer
- Dense + Dropout
- Sigmoid output layer

Hyperparameter Grid Search:
- Embedding dimension: 64 / 128
- LSTM units: 32 / 64 / 128
- Batch size: 32 / 64
- Learning rate: 0.0005 / 0.001

## Best Configuration

- Embedding Dimension: 64
- LSTM Units: 64 / 64
- Batch Size: 64
- Learning Rate: 0.001

Performance:
- Accuracy: 82.08%
- F1-Macro: 82.06%

Classification Summary:

Negative (0):
- Precision: 0.81
- Recall: 0.82

Positive (1):
- Precision: 0.84
- Recall: 0.82

Observations:
- Balanced class performance.
- Difficulty in handling implicit or ironic sentiment.
- Some lexical bias toward emotionally negative words.

---

# 4️⃣ Integrated ABSA Pipeline

Pipeline Flow:

Sentence  
→ BERT Aspect Extraction  
→ Aspect–Sentence Pairing  
→ LSTM Sentiment Classification  
→ Aspect–Sentiment Output  

The modular structure ensures separation between ATE and ASC components while maintaining automated integration.

---

# 5️⃣ Analytical Aggregation

Post-processing includes:

- Aspect frequency calculation
- Sentiment distribution per aspect
- Identification of dominant narrative elements

This enables high-level insight generation from large-scale review data.

---

# 6️⃣ Limitations

- Multi-word aspect boundary detection remains imperfect.
- Error propagation from ATE to ASC.
- LSTM limitations in sarcasm & contextual ambiguity.
- Dataset limited to English IMDb reviews.

---

# 7️⃣ Future Work

- Replace LSTM with transformer-based sentiment classifier
- Improve multi-word aspect detection
- Expand to multilingual datasets
- Apply domain adaptation techniques
