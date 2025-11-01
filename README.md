# SMS Spam Classifier

This project demonstrates the development of a **machine learning-based SMS spam detection system** that classifies text messages as **Spam** or **Ham (Not Spam)**. It includes both a **Jupyter notebook** for model building and a **Streamlit web app** for real-time message classification.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Project Workflow](#project-workflow)
4. [Data Cleaning](#1-data-cleaning)
5. [Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
6. [Data Preprocessing](#3-data-preprocessing)
7. [Model Building](#4-model-building)
8. [Streamlit Web App](#streamlit-web-app)
9. [Installation & Setup](#installation--setup)
10. [How to Run](#how-to-run)
11. [Results](#results)
12. [Future Improvements](#future-improvements)
13. [Tech Stack](#tech-stack)

---

## Overview

The goal of this project is to accurately detect **spam messages** using text classification techniques. The model is trained on a labeled dataset of SMS messages, applying NLP preprocessing, feature engineering, and multiple Naive Bayes classifiers to find the most effective model for spam detection.

The final solution integrates:

* **A Jupyter Notebook** for end-to-end model development
* **A Streamlit Web App** for user-friendly, real-time spam prediction

---

## Features

* Cleans and preprocesses raw SMS text messages
* Performs exploratory data analysis (EDA) with statistical and visual insights
* Extracts both **textual and structural features** (e.g., URLs, punctuation, uppercase ratios, etc.)
* Builds and compares multiple Naive Bayes models
* Uses **TF-IDF vectorization** for text representation
* Handles **class imbalance** using oversampling
* Provides a **Streamlit web interface** for live message prediction

---

## Project Workflow

1. Data Loading and Cleaning
2. Exploratory Data Analysis
3. Text and Feature Preprocessing
4. Model Training and Evaluation
5. Model Serialization (Pickle)
6. Streamlit Web Deployment

---

## 1. Data Cleaning

* Loaded the "Spam Email Dataset" containing SMS messages labeled as **ham** or **spam**.
* Dropped irrelevant columns: `Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4`.
* Renamed columns to `target` (label) and `text` (message).
* Encoded labels numerically (`0` for ham, `1` for spam).
* Checked and confirmed no missing values.
* Removed duplicate entries to avoid bias.

---

## 2. Exploratory Data Analysis (EDA)

* Visualized class distribution — showing dataset imbalance between spam and ham.
* Engineered new numerical features:

  * `num_characters` – length of message
  * `num_words` – number of words
  * `num_sentences` – number of sentences
* Used histograms, pair plots, and heatmaps to study feature distributions and correlations.
* Found that spam messages tend to be longer and contain more digits and special symbols.

---

## 3. Data Preprocessing

* Performed text cleaning on the `text` column:

  * Lowercasing
  * Tokenization
  * Removing non-alphanumeric characters
  * Removing stopwords and punctuation
  * Applying **stemming** using `PorterStemmer`
* Created a new column `cleaned_text` with the processed version.
* Generated **WordClouds** for spam and ham messages to visualize frequent terms.
* Identified top 30 most common words in each category and visualized with bar plots.

---

## 4. Model Building

* Transformed `cleaned_text` using **TF-IDF vectorization**.
* Combined TF-IDF vectors with engineered numeric features.
* Split the dataset using `train_test_split` with stratification.
* Addressed class imbalance using **RandomOverSampler** to duplicate minority (spam) samples.
* Trained multiple **Naive Bayes** models:

  * `GaussianNB`
  * `MultinomialNB`
  * `BernoulliNB`
  * `ComplementNB`
* Evaluated each model using:

  * `accuracy_score`
  * `precision_score`
  * `confusion_matrix`
* Chose **BernoulliNB** as the final model for its superior precision on the imbalanced dataset.
* Saved both the **TF-IDF Vectorizer** and **Trained Model** using `pickle`.

---

## Streamlit Web App

A lightweight **Streamlit application** was created to allow users to test the spam classifier interactively.

### Key Functionalities

* User enters a message in the input box.
* The app performs text preprocessing and feature extraction in real-time.
* Extracts both **text-based** (TF-IDF) and **numeric features** such as:

  * Presence of URLs, digits, punctuation, exclamation marks, uppercase ratios, phone numbers, and spammy keywords.
* Combines these features into a single sparse matrix.
* Loads the pre-trained model (`model.pkl`) and vectorizer (`vectorizer.pkl`) for prediction.
* Displays output as **Spam!!!** or **Ham**.

### Example Output

| Input Message                         | Prediction |
| ------------------------------------- | ---------- |
| "Win a free iPhone now! Click here!"  | Spam!!!    |
| "Let's catch up for dinner tomorrow." | Ham        |

---

## Installation & Setup

1. **Clone this repository**

   ```
   git clone https://github.com/<your-username>/sms-spam-classifier.git
   cd sms-spam-classifier
   ```

2. **Create and activate a virtual environment**

   ```
   python -m venv venv
   source venv/bin/activate       # for macOS/Linux
   venv\Scripts\activate          # for Windows
   ```

3. **Install dependencies**

   ```
   pip install -r requirements.txt
   ```

4. **Download NLTK data (if not already available)**

   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

---

## How to Run

**Option 1 — Run Jupyter Notebook**

```
jupyter notebook Spam_Email_Detection.ipynb
```

**Option 2 — Launch Streamlit App**

```
streamlit run app.py
```

Enter your message in the input box, click **Predict**, and view the classification.

---

## Results

* **Best Model:** Bernoulli Naive Bayes
* **Accuracy:** ~97%
* **Precision (Spam):** ~98%
* **Key Insight:**

  * Spam messages often include promotional or urgent keywords, excessive punctuation, and links.
  * Combining structural numeric features with TF-IDF text features improved accuracy and robustness.

---

## Future Improvements

* Integrate advanced NLP techniques such as:

  * **Word Embeddings (Word2Vec, GloVe)**
  * **Deep Learning models (LSTM, BERT)**
* Add language detection and multilingual support.
* Deploy app on **Streamlit Cloud**, **Render**, or **Heroku**.
* Build API endpoints for mobile integration.

---

## Tech Stack

**Languages & Libraries:**

* Python, NLTK, Scikit-learn, NumPy, Pandas, Matplotlib, Seaborn, Streamlit, SciPy, Pickle

**Machine Learning Algorithms:**

* Naive Bayes Classifiers (BernoulliNB, MultinomialNB, etc.)

**Deployment:**

* Streamlit Web App
