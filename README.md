# Sentiment Analysis using Neural Network

Classifying Google Play Store app reviews as Positive, Negative, 
or Neutral using a custom text preprocessing pipeline and a 
Keras Dense Neural Network.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Framework](https://img.shields.io/badge/Framework-Keras%20%2F%20TensorFlow-orange)
![Task](https://img.shields.io/badge/Task-Sentiment%20Analysis-green)
![Type](https://img.shields.io/badge/Type-Multi--Class%20Classification-blue)

---

## Overview

Builds a feedforward Neural Network using Keras to classify 
Google Play Store user reviews into three sentiment categories — 
Positive, Negative, and Neutral. Implements a custom text 
preprocessing pipeline that handles negation detection 
(e.g. "not good" → "not_good"), stopword removal, and 
Bag-of-Words feature extraction using CountVectorizer. 
The model is evaluated on held-out test data using accuracy.

---

## Problem Statement

User reviews on app stores contain valuable sentiment signals 
that businesses use to understand customer satisfaction and 
improve their products. This project builds an end-to-end NLP 
pipeline to automatically classify review sentiment — 
demonstrating the full workflow from raw text to a trained 
neural network classifier.

---

## Dataset

- **Name:** Google Play Store User Reviews Dataset
- **Source:** [Download from Kaggle](https://www.kaggle.com/datasets/lava18/google-play-store-apps)
- **Size:** 64,295 user reviews (after removing NaN values)
- **Classes:** Positive, Negative, Neutral (3-class classification)
- **Key Columns:** `Translated_Review` (review text), 
  `Sentiment` (target label)
- **Format:** CSV file
- **Instructions:** Download `googleplaystore_user_reviews.csv` 
  from the link above and place it in the root folder of this 
  project before running the notebook.

---

## Approach

**Data Cleaning**
- Loaded dataset using Pandas; checked shape and column types 
  with `.info()`
- Removed all rows containing NaN values using 
  `dropna(axis=0, inplace=True)`
- Printed dataset size before and after cleaning to confirm 
  rows removed

**Custom Text Preprocessing — `cleaning_data()` Function**

A custom preprocessing function was built to handle:

| Step | What it does |
|------|-------------|
| Lowercasing | Converts all words to lowercase |
| Negation detection | Detects `not`, `no`, `never`, `n't` and prefixes the next word — e.g. `"not good"` becomes `"not_good"` |
| Stopword removal | Removes common English stopwords using NLTK |
| Special character removal | Strips numbers, underscores, and non-alphabetic characters using regex `re.sub('[w_0-9]', ' ', word)` |

> The negation detection step is particularly important — without 
> it, "not good" and "good" would be treated as the same sentiment 
> signal by the model.

**Feature Extraction**
- Applied `CountVectorizer()` — Bag of Words model
- Fit on the full dataset (`vectorizer.fit(X)`) to build 
  the complete vocabulary
- Transformed training and test sets separately using 
  `.transform()`

**Label Encoding**
- Encoded string labels (Positive, Negative, Neutral) to 
  integers (0, 1, 2) using Scikit-Learn `LabelEncoder`

**Train / Test Split**
- 80% training, 20% testing using `train_test_split(test_size=0.2)`

**Neural Network Architecture**

| Layer | Type | Units | Activation |
|-------|------|-------|------------|
| Input | Dense | 100 | ReLU |
| Output | Dense | 3 | Sigmoid |

```python
model = Sequential()
model.add(Dense(units=100, activation='relu', 
                input_dim=len(vectorizer.get_feature_names_out())))
model.add(Dense(units=3, activation='sigmoid'))
```

**Training Configuration**

| Parameter | Value |
|-----------|-------|
| Loss Function | Sparse Categorical Cross-Entropy |
| Optimiser | Adam |
| Epochs | 2 |
| Metric | Accuracy |

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | [add your value — e.g. 0.87] |
| Epochs Trained | 2 |
| Classes | Positive, Negative, Neutral |
| Vocabulary Size | [add value from `len(vectorizer.vocabulary_)`] |

> Run the notebook and replace `[add your value]` with the 
> accuracy printed by `model.evaluate()` in the final cell.


---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/OyelolaIbrahim/sentiment-analysis-neural-network.git
cd sentiment-analysis-neural-network

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset from Kaggle and place 
#    googleplaystore_user_reviews.csv in the root folder

# 4. Run the notebook
jupyter notebook sentiment_analysis_nn.ipynb
```


- The model trains in only 2 epochs — increasing epochs 
  with an early stopping callback could improve accuracy 
  significantly
