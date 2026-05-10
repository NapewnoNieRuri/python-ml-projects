import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('heart.csv')

def preprocess_data(df):
    df = df.copy()

    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 44, 59, 77], labels=False, include_lowest=True)

    df['Sex'] = df['Sex'].map({'F': 1, 'M': 0})
    df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})
    df['ST_Slope'] = df['ST_Slope'].map({'Up': 2, 'Flat': 1, 'Down': 0})

    return df

df = preprocess_data(df)

df = df.dropna()

X = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

categorical_features = ['ChestPainType', 'RestingECG']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', XGBClassifier(
        eval_metric='logloss',
        random_state=42
    ))
])

param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 5],
    'model__learning_rate': [0.05, 0.1],
    'model__subsample': [0.8, 1.0]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='recall',
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

predictions = best_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion matrix:\n", confusion_matrix(y_test, predictions))
print("\nReport:\n", classification_report(y_test, predictions))

def plot_model(matrix):
    plt.figure(figsize=(6,5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt='d',
        xticklabels=['NoHeartDisease', 'HeartDisease'],
        yticklabels=['NoHeartDisease', 'HeartDisease']
    )
    plt.title('Confusion matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

matrix = confusion_matrix(y_test, predictions)
plot_model(matrix)