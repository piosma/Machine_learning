import pandas as pd
from sklearn.model_selection import train_test_split #Biblioteka do podziału danych na zbiór treningowy i testowy
from sklearn.ensemble import RandomForestClassifier #Algorytm machine learning stosowany do klasyfikacji.
from sklearn.metrics import accuracy_score # Accuracy_score służy do porównania prognozy z rzczywistymi wynikami
import joblib  # Do zapisywania modelu

# Wczytanie przetworzonych danych z wcześniej utworzonego pliku
cocktail_info = pd.read_csv("cocktail_processed_data.csv")

# Dane wejściowe modelu, wykorzystujemy do nauki. Pozostają tylko kolumny ze składnikami.
X = cocktail_info.drop( columns = ["Drink", "Category"])

# Nasza zmienna w którą celujemy (chcemy przwidzieć na podstawie składników)
y = cocktail_info["Category"]

# Podział na zestawy treningowe i testowe (80% trening, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trenowanie modelu Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)  # Uczenie modelu

# Przewidywanie na zbiorze testowym
y_pred = model.predict(X_test)

# Ocena skuteczności
accuracy = accuracy_score(y_test, y_pred)
print(f"Dokładność modelu: {accuracy:.2f}")

# Zapisanie modelu do pliku
joblib.dump(model, "cocktail_model.pkl")
print("Model zapisany jako cocktail_model.pkl")
