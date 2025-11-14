📌 Smishing (SMS Phishing) Detection Using Machine Learning

This project implements a Machine Learning–based system to detect smishing (SMS phishing) messages using classical NLP techniques such as Bag-of-Words, TF-IDF, and models like Logistic Regression, Naive Bayes, and SVM.
The goal is to build a fast, lightweight, and highly interpretable solution that can classify SMS messages as phishing or legitimate.

🚀 Features

Data Cleaning & Text Preprocessing
Lowercasing, URL removal, punctuation cleaning, stopword removal, tokenization, etc.

Feature Engineering

Bag-of-Words

TF-IDF

n-grams (1-gram and 2-gram)

Optional: Feature selection using chi-square

ML Model Training & Evaluation
Models used:

Logistic Regression

Naive Bayes

Support Vector Machine

Random Forest (for comparison)

Hyperparameter Tuning
GridSearchCV for best parameters

Evaluation Metrics

Accuracy

Precision

Recall

F1-score

Confusion Matrix

ROC-AUC

Production-Ready Pipeline
Text preprocessor → TF-IDF vectorizer → final ML classifier

🛠️ Technologies Used

Python

Scikit-learn

Pandas & NumPy

NLTK

Matplotlib / Seaborn

Jupyter Notebook

📂 Project Structure
Smishing-Detection-ML/
│── Smishing-ML.ipynb          # ML pipeline with training & evaluation
│── SMSSmishCollection.txt     # Dataset
│── models/
│     └── smishing_model.pkl   # Saved ML model
│── vectorizer/
│     └── tfidf.pkl            # Saved TF-IDF vectorizer
│── README.md                  # Project documentation (this file)

🔧 How to Run
1. Install dependencies
pip install -r requirements.txt

2. Open the notebook
jupyter notebook Smishing-ML.ipynb

3. Train & Test

Execute all cells to preprocess data, train the model, and evaluate results.

📈 Results

Best model: Logistic Regression (TF-IDF + bigrams)

Achieved:

Accuracy: ~97–99%

Precision: High (important for phishing detection)

Excellent recall on phishing messages
