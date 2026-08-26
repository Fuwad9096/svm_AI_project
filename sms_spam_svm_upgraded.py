#!/usr/bin/env python3
"""
SMS Spam Classification using SVM (Upgraded)
Dataset: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

Converted from SMS_Spam_SVM_Upgraded.ipynb
"""


# # SMS Spam Classification using SVM (Upgraded)
# Dataset: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download
#
# **Upgrades in this version:**
# - Hand-crafted spam-signal features (punctuation, digits, capitalization, URLs, bait words)
# - Combined TF-IDF + numeric features via `ColumnTransformer`
# - Class-imbalance handling (`class_weight='balanced'`)
# - Expanded grid search, scored on F1 instead of accuracy
# - Confusion matrix + Precision-Recall curve (via `CalibratedClassifierCV`)
# - Explainability: top spam/ham indicative features
# - Error analysis on misclassified messages
# - Model persistence with `joblib`


# Install (if needed) — run this line in your terminal, not as Python code:
# pip install wordcloud nltk scikit-learn


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from wordcloud import WordCloud
import re, string, nltk
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, roc_curve, auc,
    precision_recall_curve, PrecisionRecallDisplay)

nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ## Upload dataset (`spam.csv`) downloaded from Kaggle.


# Make sure SMS.csv (the Kaggle SMS Spam Collection file) is in the same
# directory as this script, or update the path in pd.read_csv() below.


df = pd.read_csv("SMS.csv", encoding="latin-1")

df = df[['v1', 'v2']]

df.columns = ['label', 'message']


print(df.shape)
print(df.info())
print(df.isnull().sum())
print("Duplicates:", df.duplicated().sum())
print(df['label'].value_counts())


# ## Exploratory Data Analysis


df['char_count'] = df['message'].str.len()
df['word_count'] = df['message'].str.split().str.len()

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.countplot(data=df, x='label', ax=ax[0])
sns.histplot(data=df, x='word_count', hue='label', bins=30, ax=ax[1])
plt.tight_layout()
plt.show()


spam = ' '.join(df[df.label == 'spam'].message)
ham = ' '.join(df[df.label == 'ham'].message)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(WordCloud(background_color='white').generate(spam))
plt.axis('off'); plt.title("Spam")
plt.subplot(1, 2, 2)
plt.imshow(WordCloud(background_color='white').generate(ham))
plt.axis('off'); plt.title("Ham")
plt.show()


# ## Preprocessing


# Load the list of English stopwords (e.g., "the", "is", "and")
stop_words = set(stopwords.words('english'))

# Create a WordNet lemmatizer object to convert words to their base form
lemm = WordNetLemmatizer()

# Define a function to clean and preprocess SMS text
def clean_text(text):

    # Convert all characters to lowercase
    text = text.lower()

    # Remove URLs (e.g., http://example.com or www.example.com)
    text = re.sub(r'http\S+|www\S+', ' ', text)

    # Remove email addresses (e.g., abc@gmail.com)
    text = re.sub(r'\S+@\S+', ' ', text)

    # Remove all numeric digits from the text
    text = re.sub(r'\d+', ' ', text)

    # Remove punctuation marks such as !, ?, ., ,, :, etc.
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Split the sentence into words,
    # remove stopwords,
    # and convert each word to its base (lemma) form
    tokens = [
        lemm.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    ]

    # Join the cleaned words back into a single string
    return ' '.join(tokens)

# Remove duplicate SMS messages from the dataset
df = df.drop_duplicates()

# Apply the clean_text() function to every message
# and store the cleaned text in a new column named 'clean_message'
df['clean_message'] = df['message'].apply(clean_text)

# Convert categorical labels into numerical values
# ham  -> 0
# spam -> 1
df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

# Display the first five rows of the processed dataset
df.head()


# ## Feature Engineering (NEW)
# `clean_text` strips punctuation, digits, and case — but for spam detection these are strong signals (`!!!`, phone numbers/prize amounts, ALL CAPS). Instead of discarding them, we extract them as separate numeric features computed on the **raw** message, before cleaning.


def extract_spam_features(text: str) -> dict:
    """Extract hand-crafted spam-signal features from a RAW (uncleaned) SMS message."""
    text = str(text)
    letters = [c for c in text if c.isalpha()]

    return {
        'exclamation_count': text.count('!'),
        'digit_count': sum(c.isdigit() for c in text),
        'uppercase_ratio': (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0,
        'message_length': len(text),
        'has_url': int(bool(re.search(r'http\S+|www\S+', text, re.IGNORECASE))),
        'has_money_or_bait_word': int(bool(re.search(
            r'[\u00a3$\u20ac]|\bfree\b|\bwin\b|\bwon\b|\bprize\b|\burgent\b',
            text, re.IGNORECASE)))
    }

# Apply to RAW messages (df['message']), NOT the cleaned text
feature_df = pd.DataFrame(df['message'].apply(extract_spam_features).tolist())
df = pd.concat([df.reset_index(drop=True), feature_df], axis=1)

numeric_features = ['exclamation_count', 'digit_count', 'uppercase_ratio',
                     'message_length', 'has_url', 'has_money_or_bait_word']

df[['message', 'label'] + numeric_features].head()


# ## Train/Test Split


X = df[['clean_message'] + numeric_features]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(X_train.shape)
print(X_test.shape)


# ## SVM Model (UPGRADED)
# Combines TF-IDF text features with the numeric spam-signal features via `ColumnTransformer`, and uses `class_weight='balanced'` since spam is only ~13% of the dataset.


preprocessor = ColumnTransformer(transformers=[
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True),
     'clean_message'),
    ('numeric', StandardScaler(), numeric_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('svm', LinearSVC(class_weight='balanced'))
])

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("Recall:", recall_score(y_test, pred))
print("F1:", f1_score(y_test, pred))
print(classification_report(y_test, pred))


# ## Hyperparameter Tuning (UPGRADED)
# Expanded grid (ngram range, max_features, C, class_weight), scored on **F1** rather than accuracy since accuracy is misleading on this imbalanced dataset.


param_grid = {
    'preprocessor__tfidf__ngram_range': [(1, 1), (1, 2)],
    'preprocessor__tfidf__max_features': [3000, 5000],
    'svm__C': [0.1, 1, 10],
    'svm__class_weight': [None, 'balanced']
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best CV F1 Score:", grid.best_score_)


# ## Cross Validation


scores = cross_val_score(grid.best_estimator_, X, y, cv=5, scoring='f1')
print(scores)
print("Mean F1:", scores.mean())


# ## Evaluation: Confusion Matrix & Precision-Recall Curve (NEW)
# `LinearSVC` has no `predict_proba` (it outputs a decision-boundary distance, not a calibrated probability), so we wrap it in `CalibratedClassifierCV` to get probability scores for the PR curve.


best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['ham', 'spam'])
plt.title("Confusion Matrix")
plt.show()

calibrated = CalibratedClassifierCV(best_model, cv=5)
calibrated.fit(X_train, y_train)
y_scores = calibrated.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
PrecisionRecallDisplay(precision=precision, recall=recall).plot()
plt.title("Precision-Recall Curve")
plt.show()


# ## Explainability (NEW)
# Since `LinearSVC` is a linear model, its coefficients directly tell us which features push a prediction toward spam vs. ham — this is why the model flags what it flags.


tfidf_feature_names = best_model.named_steps['preprocessor'].named_transformers_['tfidf'].get_feature_names_out()
all_feature_names = list(tfidf_feature_names) + numeric_features

coefs = best_model.named_steps['svm'].coef_[0]
top_spam_idx = coefs.argsort()[-15:][::-1]
top_ham_idx = coefs.argsort()[:15]

print("Top spam-indicative features:")
for i in top_spam_idx:
    print(f"  {all_feature_names[i]}: {coefs[i]:.3f}")

print("\nTop ham-indicative features:")
for i in top_ham_idx:
    print(f"  {all_feature_names[i]}: {coefs[i]:.3f}")


# ## Error Analysis (NEW)
# Inspecting misclassified messages to understand what the model still gets wrong.


errors = X_test.copy()
errors['true_label'] = y_test.values
errors['predicted'] = y_pred
errors = errors[errors['true_label'] != errors['predicted']]

print(f"Total misclassified: {len(errors)} out of {len(X_test)}")
errors[['clean_message', 'true_label', 'predicted']].head(10)


# ## Save Model (NEW)


joblib.dump(best_model, 'sms_spam_model.pkl')
print("Model saved to sms_spam_model.pkl")


# ## Prediction Function (UPGRADED)
# Now also computes the spam-signal features for new input text.


model = best_model

def predict_sms(text):
    features = extract_spam_features(text)
    clean = clean_text(text)
    input_df = pd.DataFrame([{'clean_message': clean, **features}])
    pred = model.predict(input_df)[0]
    return "Spam" if pred == 1 else "Ham"

print(predict_sms("Congratulations! You have won a free iPhone. Click here now."))
print(predict_sms("Hi, are we meeting tomorrow at 3 PM?"))

