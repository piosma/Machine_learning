import pandas as pd

#Wstepna analiza problemu i zbioru danych EDA
#Wczytanie danych z pliku .json
cocktail_info = pd.read_json("cocktail_dataset.json")

#Sprawdzamy za pomocą funkcji .head() czy wstępne wczytanie przebiegło pomyślnie (5 pierwszych wierszy)
print(cocktail_info.head())

# Rozpoznanie typów danych wraz z wartościami "niepustymi"
print(cocktail_info.info())

# Podzielenie na kategorie drinków i sprawdzenie ich liczby
print(cocktail_info["category"].value_counts())

# Sprawdzamy gdzie mamy braki w danych (tagi 99 pustych miejsc)
print(cocktail_info.isnull().sum())

#Podgląd, jak wyglądają składniki w jednym drinku
print(cocktail_info["ingredients"].iloc[0])  # Zobaczymy listę składników pierwszego drinka

# Konwersja składników na listę nazw
cocktail_info["ingredients"] = cocktail_info["ingredients"].apply(lambda x: [i["name"] for i in x])

# Sprawdzenie efektu
print(cocktail_info["ingredients"].head())

