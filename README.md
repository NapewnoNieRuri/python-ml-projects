Witam w moim repozytorium! Zawiera ono zbiór projektów z dziedziny prostego uczenia maszynowego (Machine Learning) oraz analizy danych (Data Science).

## 📂 Przegląd Projektów

### 1. Wykrywanie Oszustw Kredytowych (Fraud Detection)
Celem projektu jest klasyfikacja transakcji kartowych jako oszukańcze (fraud) lub poprawne. Zbiór danych cechuje się ogromnym niezbalansowaniem klas.
* **Model:** Pipeliny scikit-learn (`StandardScaler` + `SGDClassifier`).
* **Kluczowe kroki:** Skalowanie cech, dostosowanie wag klas (`class_weight='balanced'`), optymalizacja progu decyzyjnego (threshold).
* **Ewaluacja:** Klasyfikacja za pomocą miar wrażliwych na niezbalansowanie (Recall, Precision, Roc-AUC, Confusion Matrix).

### 2. Przewidywanie Niewydolności Serca (Heart Failure Prediction)
Projekt skupia się na przewidywaniu chorób serca na podstawie danych medycznych pacjentów (np. wiek, płeć, poziom cholesterolu, EKG spoczynkowe).
* **Model:** `XGBClassifier` (XGBoost).
* **Kluczowe kroki:** Kategoryzacja ciągłych zmiennych (binowanie wieku), Feature Engineering, transformacja zmiennych kategorycznych (One-Hot Encoding za pomocą `ColumnTransformer`).
* **Optymalizacja:** Strojenie hiperparametrów za pomocą `GridSearchCV` w celu maksymalizacji wskaźnika **Recall** (czułość modelu to w tym przypadku najważniejsza metryka, aby nie przeoczyć chorego pacjenta).

### 3. Przewidywanie Cen Laptopów (Laptop Price Prediction)
Regresyjny model uczenia maszynowego szacujący cenę laptopa na podstawie jego specyfikacji technicznej (m.in. procesor, RAM, rozdzielczość ekranu, pamięć, waga).
* **Model:** Zespół modeli wykorzystujący regularyzację - `ElasticNet`.
* **Kluczowe kroki:** Rozbudowany Feature Engineering – ekstrakcja danych z ciągów znaków (np. wyciąganie wagi, rozmiaru RAM, rozdzielczości X i Y z tekstu), Target Encoding dla nazw produktów.
* **Optymalizacja:** `GridSearchCV` dla doboru optymalnych współczynników kary (alpha oraz l1_ratio).
* **Ewaluacja:** Miary `R2 Score` oraz Średni Błąd Bezwzględny (`MAE`). W modelu zastosowano wizualizacje dopasowania predykcji do wartości rzeczywistych.

---

## 🚀 Jak uruchomić projekty?

Aby uruchomić poszczególne skrypty na swoim środowisku lokalnym:

1. Sklonuj repozytorium:
   ```bash
   git clone [https://github.com/TwojLogin/NazwaTwojegoRepo.git](https://github.com/TwojLogin/NazwaTwojegoRepo.git)
