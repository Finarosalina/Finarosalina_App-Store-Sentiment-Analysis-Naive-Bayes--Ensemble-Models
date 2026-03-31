# 💬 App Store Sentiment Analysis — Naive Bayes & Ensemble Models

**Binary sentiment classification of Google Play Store reviews using Naive Bayes variants, VotingClassifier, and XGBoost, with systematic model comparison and hyperparameter optimization.**

---

## Overview

This project builds a sentiment classifier to predict whether a Google Play Store review is positive or negative (polarity: 0 = negative, 1 = positive). Three Naive Bayes variants are evaluated and compared, followed by ensemble methods and XGBoost, with honest analysis of where each model succeeds and fails.

A key methodological insight emerges: for text classification with count-based features, MultinomialNB outperforms GaussianNB and BernoulliNB — despite BernoulliNB being theoretically appealing for binary outputs.

---

## Dataset

- **Source:** [4Geeks Academy Naive Bayes Tutorial](https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews.csv)
- **Records:** 891 reviews (after preprocessing)
- **Features:** Review text (vectorized with CountVectorizer)
- **Target:** Polarity — 0 (negative) / 1 (positive)
- **Class distribution:** ~70% negative / ~30% positive

---

## Methodology

**1. Preprocessing**
- Removed `package_name` column
- Lowercased and stripped review text
- CountVectorizer with English stopword removal
- Train/test split: 80/20, random_state=42

**2. Models Evaluated**

| Model | Train Accuracy | Test Accuracy | Notes |
|---|---|---|---|
| MultinomialNB (baseline) | 0.961 | 0.816 | Best overall for count features |
| GaussianNB | 0.986 | 0.804 | Overfits — dense matrix required |
| BernoulliNB | 0.920 | 0.771 | Weaker on frequency data |
| VotingClassifier (MNB + RF) | 0.971 | 0.810 | No improvement over MNB alone |
| MultinomialNB optimized (α=0.5) | 0.968 | **0.827** | Best test performance |
| XGBoost baseline | 0.965 | 0.810 | High overfitting risk |
| XGBoost optimized (GridSearchCV) | 0.885 | 0.777 | Reduced overfitting, lower accuracy |

**3. Hyperparameter Optimization**
- MultinomialNB: RandomizedSearchCV over alpha ∈ [0.1, 0.5, 1.0, 2.0, 5.0] → best alpha = 0.5
- XGBoost: GridSearchCV over 108 combinations (n_estimators, learning_rate, max_depth, subsample, colsample_bytree) → best params selected but did not improve test accuracy

**4. Final Model**
Optimized MultinomialNB (alpha=0.5) — best test accuracy (82.7%) with lowest computational cost.

---

## Key Finding

BernoulliNB was initially expected to perform best given the binary classification target. However, the feature space is count-based (word frequencies), not binary presence/absence — making MultinomialNB the more appropriate and better-performing choice. This highlights the importance of matching model assumptions to feature distribution, not just to the output variable type.

XGBoost with GridSearchCV (324 fits) reduced overfitting but delivered lower test accuracy than the simple optimized MultinomialNB — a case where a simpler model wins on both performance and efficiency.

---

## Results Summary

**Final model — MultinomialNB (alpha=0.5):**

| Metric | Class 0 (Negative) | Class 1 (Positive) | Overall |
|---|---|---|---|
| Precision | 0.86 | 0.73 | — |
| Recall | 0.90 | 0.66 | — |
| F1-score | 0.88 | 0.69 | — |
| Accuracy | — | — | **0.827** |

---

## Tech Stack

- **NLP** — scikit-learn CountVectorizer
- **Models** — MultinomialNB, GaussianNB, BernoulliNB, RandomForestClassifier, XGBClassifier
- **Ensemble** — VotingClassifier (soft voting)
- **Optimization** — RandomizedSearchCV, GridSearchCV
- **Persistence** — pickle

---

## Project Structure

```
Finarosalina_Bayes_bueno_MlL/
├── src/
│   ├── explore.ipynb                          # Full analysis notebook
│   └── app.py                                 # Exported pipeline script
├── models/
│   └── modelo_voting_classifier_opt.pkl       # Saved optimized VotingClassifier
├── data/
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
└── README.md
```

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/Finarosalina/Finarosalina_Bayes_bueno_MlL.git
cd Finarosalina_Bayes_bueno_MlL

# Install dependencies
pip install pandas scikit-learn xgboost

# Run the notebook
jupyter notebook src/explore.ipynb
```

---

## Key Learnings

- Model selection should be driven by feature distribution, not just output type — CountVectorizer output is multinomial, not binary
- GaussianNB requires dense matrix conversion and overfits significantly on sparse text features
- VotingClassifier did not improve over MultinomialNB alone — ensemble methods are not always better than a well-tuned single model
- XGBoost with extensive GridSearchCV (108 combinations, 324 fits) failed to outperform a simple Naive Bayes — a reminder that computational cost does not equal performance
- Laplace smoothing (alpha) in MultinomialNB is a high-impact, low-cost hyperparameter worth tuning first

---

## Related Projects

- [🔗 URL Spam Detection (NLP + SVM)](https://github.com/Finarosalina/Finarosalina_NLP_DL) — TF-IDF + SVM pipeline
- [🍷 KNN Wine Quality Classifier](https://github.com/Finarosalina/Finarosalina_KNN_BUENO_ML_WINE) — supervised classification with imbalanced data
- [🌍 Air Quality & Mortality](https://github.com/ullaom/air-quality-politics) — Random Forest with Streamlit deployment

---

*Part of the 4Geeks Academy Data Science & ML program portfolio.*
