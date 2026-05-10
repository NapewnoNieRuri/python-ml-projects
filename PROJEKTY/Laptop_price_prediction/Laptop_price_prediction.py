import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.compose import ColumnTransformer

df = pd.read_csv('laptop_price.csv', sep=',', header=0, encoding='latin-1')


def preprocess_data(df):
    product_means = df.groupby('Product')['Price_euros'].mean()
    df['Product_Target_Enc'] = df['Product'].map(product_means)

    df.drop(['laptop_ID', 'Product'], axis=1, inplace=True)
    df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
    df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)

    res = df['ScreenResolution'].str.extract(r'(\d+)x(\d+)')
    df['X_res'] = res[0].astype(int)
    df['Y_res'] = res[1].astype(int)
    df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
    df['Ips'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS Panel' in x else 0)
    df.drop('ScreenResolution', axis=1, inplace=True)

    df['Cpu_freq'] = df['Cpu'].str.extract(r'(\d+\.?\d*)GHz').astype(float)
    df['Cpu'] = df['Cpu'].str.replace(r'\s*\d+\.?\d*GHz', '', regex=True).str.strip()

    df['Gpu'] = df['Gpu'].apply(lambda x: x.split()[0])
    df['Memory'] = df['Memory'].astype(str).str.replace('GB', '').str.replace('TB', '000')
    df['SSD'] = df['Memory'].apply(lambda x: sum([int(s) for s in re.findall(r'(\d+)\s*SSD', x)]) if 'SSD' in x else 0)
    df['HDD'] = df['Memory'].apply(lambda x: sum([int(s) for s in re.findall(r'(\d+)\s*HDD', x)]) if 'HDD' in x else 0)

    df.drop('Memory', axis=1, inplace=True)
    return df


data = preprocess_data(df)
X = data.drop(columns=['Price_euros'])
y = data['Price_euros']

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

features = ['Company', 'TypeName', 'Cpu', 'Gpu', 'OpSys']
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features)
], remainder='passthrough')

X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

param_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
    'l1_ratio': [0.1, 0.5, 0.9, 0.99]
}

grid_search = GridSearchCV(ElasticNet(random_state=42, max_iter=10000), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Najlepsze parametry: {grid_search.best_params_}")

y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'R2 score: {r2:.4f}')
print(f'Średni błąd (MAE): {mae:.2f} Euro')

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
plt.title('Ceny rzeczywiste vs Przewidziane')
plt.xlabel('Cena rzeczywista [Euro]')
plt.ylabel('Cena przewidziana [Euro]')

plt.subplot(1, 2, 2)
ohe_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(features)
all_cols = list(ohe_cols) + [col for col in X.columns if col not in features]
importances = pd.Series(best_model.coef_, index=all_cols).abs().sort_values(ascending=False).head(10)

sns.barplot(x=importances.values, y=importances.index, palette='viridis')
plt.title('Top 10 najważniejszych cech dla ceny')
plt.xlabel('Siła wpływu (współczynnik)')

plt.tight_layout()
plt.show()
