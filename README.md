# Multi-Model Sentiment Analysis on Customer Reviews

End-to-end NLP pipeline for classifying customer sentiment (positive / neutral / negative) from bike rental and car reviews. The project compares traditional machine learning classifiers against a transformer-based approach, and critically examines a case of near-perfect accuracy to surface a data leakage issue.

## Project Overview

| Step | Description |
|------|-------------|
| Text preprocessing | Lowercasing, punctuation removal, tokenization, stopword removal, lemmatization (NLTK) |
| Feature engineering | TF-IDF vectorization (top 5,000 features) |
| Classical models | Multinomial Naive Bayes and Logistic Regression with 5-fold stratified cross-validation |
| Transformer model | DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`) via HuggingFace Pipelines |
| Leakage detection | Shuffle test to validate suspiciously perfect classical model results |

## Datasets

| File | Description |
|------|-------------|
| `bike_rental_reviews.csv` | 50,000 bike rental reviews with labelled sentiment (positive / neutral / negative) |
| `Car_Reviews_Database.csv` | Car reviews (unlabelled — used for BERT inference) |

## Key Results

- **Naive Bayes & Logistic Regression** achieved 100% accuracy across all 5 folds — an immediate red flag.
- A **shuffle test** (randomly permuting labels) dropped accuracy to ~0.33 (chance level), confirming the bike rental data is trivially separable — likely due to lexical overlap between labels and review text.
- **DistilBERT** produced realistic and robust sentiment predictions on both datasets without relying on the training labels, demonstrating the value of pretrained transformers when data quality is uncertain.

## Takeaway

> Perfect accuracy in sentiment analysis is a red flag, not a success. Understanding *why* a model performs well is as important as the metric itself.

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook sentiment_analysis.ipynb
```

### Requirements

```
numpy
pandas
matplotlib
seaborn
nltk
scikit-learn
transformers
torch
ftfy
```

## File Structure

```
repo_nlp_sentiment_reviews/
├── sentiment_analysis.ipynb   # Full notebook: preprocessing → models → leakage analysis
├── bike_rental_reviews.csv    # Bike rental reviews with sentiment labels
├── Car_Reviews_Database.csv   # Car reviews (used for BERT inference)
└── README.md
```

## Methods at a Glance

```
Raw text
  → Lowercase + remove punctuation
  → Tokenize (NLTK)
  → Remove stopwords
  → Lemmatize (WordNetLemmatizer)
  → TF-IDF (5,000 features)
  → Naive Bayes / Logistic Regression (5-fold CV)
  → Shuffle test (leakage check)
  → DistilBERT (zero-shot inference)
```
