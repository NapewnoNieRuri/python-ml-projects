import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('creditcard.csv')

df.drop('Time', axis=1, inplace=True)

X = df.drop(columns=['Class'])
y = df['Class']


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = make_pipeline(
    StandardScaler(),
    SGDClassifier(
        loss='log_loss',
        class_weight='balanced',
        max_iter=5000,
        tol=1e-4,
        random_state=42
    )
)


model.fit(X_train, y_train)


probs = model.predict_proba(X_test)[:, 1]


threshold = 0.3
predicted = (probs > threshold).astype(int)


print("\nClassification report:\n")
print(classification_report(y_test, predicted))

matrix = confusion_matrix(y_test, predicted)

print("\nConfusion matrix:\n")
print(matrix)

print("\nROC AUC:", roc_auc_score(y_test, probs))
print("PR AUC:", average_precision_score(y_test, probs))


def plot_matrix(matrix):
    plt.figure(figsize=(5,4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt='g',
        xticklabels=['NotFraud', 'Fraud'],
        yticklabels=['NotFraud', 'Fraud']
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


plot_matrix(matrix)