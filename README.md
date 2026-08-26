# SMS Spam Classification using SVM

A binary text classifier that labels SMS messages as **spam** or **ham** (not spam), built with TF-IDF text features, hand-crafted spam-signal features, and a Linear SVM.

## Dataset

[SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) (Kaggle/UCI) — 5,572 SMS messages labeled `ham` or `spam` (~87% ham, ~13% spam).

## Pipeline Overview

1. **EDA** — class balance, message length distribution, word clouds for spam vs. ham.
2. **Text preprocessing** — lowercasing, URL/email/digit removal, punctuation stripping, stopword removal, lemmatization (`clean_text`).
3. **Feature engineering** — hand-crafted numeric features extracted from the *raw* message: exclamation count, digit count, uppercase ratio, message length, URL presence, presence of money/bait words (`free`, `win`, `£`, `$`, etc.).
4. **Feature combination** — `ColumnTransformer` merges TF-IDF (unigrams + bigrams) with the scaled numeric features into a single feature matrix.
5. **Model** — `LinearSVC` with `class_weight='balanced'` to account for class imbalance.
6. **Hyperparameter tuning** — `GridSearchCV` over TF-IDF n-gram range, `max_features`, SVM `C`, and `class_weight`, scored on **F1** (not accuracy, which is misleading on imbalanced data).
7. **Evaluation** — precision/recall/F1, confusion matrix, and a precision-recall curve (via `CalibratedClassifierCV`, since `LinearSVC` has no native `predict_proba`).
8. **Explainability** — inspects the linear model's coefficients to show the top spam- and ham-indicative features.
9. **Error analysis** — lists misclassified test messages for manual inspection.
10. **Persistence** — saves the trained pipeline with `joblib` for reuse.

## Why the hand-crafted features matter

Standard text cleaning strips digits, punctuation, and case — but for spam detection these are strong signals in their own right (`!!!`, phone numbers/prize amounts, ALL CAPS shouting). Rather than discarding them, this pipeline extracts them separately from the **raw** message (before cleaning) and feeds them to the model alongside the TF-IDF text features.

## Setup

```bash
pip install pandas numpy matplotlib seaborn wordcloud nltk scikit-learn joblib
```

The notebook downloads required NLTK data (`stopwords`, `wordnet`) automatically on first run.

## Usage

1. Open `SMS_Spam_SVM_Upgraded.ipynb` in Jupyter/Colab.
2. Upload `spam.csv` (or `SMS.csv`, matching the load cell) when prompted.
3. Run all cells top to bottom.
4. Use the trained model on new messages:

```python
predict_sms("Congratulations! You have won a free iPhone. Click here now.")
# -> "Spam"

predict_sms("Hi, are we meeting tomorrow at 3 PM?")
# -> "Ham"
```

## Files

| File | Description |
|---|---|
| `SMS_Spam_SVM_Upgraded.ipynb` | Full notebook: EDA, preprocessing, feature engineering, model training, tuning, evaluation, explainability |
| `sms_spam_model.pkl` | Saved trained pipeline (generated after running the notebook) |

## Metrics reported

- Accuracy, Precision, Recall, F1 (test set)
- 5-fold cross-validated F1
- Confusion matrix
- Precision-recall curve
- Top 15 spam-indicative and ham-indicative features (by SVM coefficient weight)

## Possible extensions

- Character-level n-grams (`analyzer='char_wb'`) to catch obfuscated spam (e.g., "fr33", "V1@gra")
- Compare `LinearSVC` against `ComplementNB` / `LogisticRegression` on PR-AUC
- Deploy via a small Flask/Gradio demo
