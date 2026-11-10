# Credit Card Fraud Detection

Binary classification project to detect fraudulent credit card transactions.

## Project Overview

This project aims to identify fraudulent transactions in a highly imbalanced credit card dataset. The positive class (fraud) represents a very small fraction of the data, so evaluation focuses on **F1-score**, **Recall**, and **Balanced Accuracy** rather than overall accuracy.

## Dataset

| Item | Details |
|------|---------|
| Source | Anonymized credit card transactions |
| Train size | 170,884 rows × 31 columns |
| Features | `Time`, `V1`–`V28`, `Amount` |
| Target | `Class` (0 = Legitimate, 1 = Fraud) |
| Feature type | `V1`–`V28` are PCA-transformed features (original features hidden for confidentiality) |
| Missing values | None |
| Class imbalance | Highly imbalanced (fraud is the minority class) |

### Feature Description

- **Time**: Seconds elapsed between each transaction and the first transaction
- **V1 – V28**: Principal components obtained with PCA (anonymized)
- **Amount**: Transaction amount
- **Class**: Target variable (0 = normal, 1 = fraud)

The data is already split into:
- `data/train.csv`
- `data/val.csv`

## Project Structure

```
├── data/
│   ├── train.csv
│   └── val.csv
├── credit_fraud.ipynb              # Exploratory Data Analysis
├── credit_fraud_utils_data.py      # Data loading, scaling, resampling
├── credit_fraud_utils_eval.py      # Threshold optimization
├── credit_fraud_train.py           # Random Forest training
└── credit_fraud_train_log.py       # Logistic Regression training
```

## Approach

1. **Exploratory Data Analysis**
   - Class distribution
   - Missing values and duplicates check
   - Feature distributions (`Amount`, `Time`)
   - Correlation analysis

2. **Preprocessing**
   - Feature / target split
   - Standard scaling (for Logistic Regression)
   - SMOTE oversampling (`sampling_strategy=0.2`) to partially address class imbalance

3. **Modeling**
   - **Logistic Regression** (with scaling + SMOTE)
   - **Random Forest** (with SMOTE + `class_weight="balanced"`)
   - Hyperparameter tuning via `GridSearchCV` (F1 scoring)
   - Decision threshold optimization on the validation set to maximize F1-score

4. **Evaluation Metrics**
   - Classification Report (Precision, Recall, F1)
   - Balanced Accuracy
   - Confusion Matrix
   - Best threshold selected from Precision-Recall curve

## How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn

# Run Exploratory Analysis
jupyter notebook credit_fraud.ipynb

# Train Logistic Regression
python credit_fraud_train_log.py

# Train Random Forest
python credit_fraud_train.py
```

## Notes

- All transformers and resampling are applied on the training set only.
- Threshold tuning is performed on the validation set.
- The final reported metrics are computed on the validation set after threshold optimization.
