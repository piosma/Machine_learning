import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# ️Wczytanie danych
cocktail_info = pd.read_json("cocktail_dataset.json")

# Konwersja kolumny 'ingredients' (z listy słowników na listę nazw składników)
cocktail_info["ingredients"] = cocktail_info["ingredients"].apply(lambda x: [ingredient["name"] for ingredient in x])

# One-hot encoding. Tworzenie kolumn  z 0 i 1 (1 jeśli dana składnik występuje w drinku, 0 jesli go nie ma)
mlb = MultiLabelBinarizer()
ingredient_matrix = mlb.fit_transform(cocktail_info["ingredients"])

# Tworzenie nowej tabeli z zakodowanymi składnikami
Ingredient_Matrix = pd.DataFrame(ingredient_matrix, columns = mlb.classes_, index = cocktail_info.index)

# Dodanie kolumn z nazwą drinka oraz jego kategorią
Ingredient_Matrix.insert(0, "Drink", cocktail_info["name"])
Ingredient_Matrix.insert(1, "Category", cocktail_info["category"])

# Zapis przekształconych danych do pliku csv
Ingredient_Matrix.to_csv("cocktail_processed_data.csv", index = False)

# Podgląd pierwszych 5 wierszy
print(Ingredient_Matrix.head())

