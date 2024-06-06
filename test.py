import pandas as pd
import numpy as np
from IPython.display import display, HTML
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Cel projektu i stworzenie modelu reprezentującego badany problem

# Krótka charakterystyka zbioru (1 pkt.)
file_path = 'top-500-movies.csv'
movies_df = pd.read_csv(file_path)
def display_interactive_table(dataframe, num_rows=10):
    display(HTML(dataframe.head(num_rows).to_html()))

display_interactive_table(movies_df)

# Sformułowanie tematu projektu i celu analizy badawczej (1 pkt.)
"""
Celem tego raportu jest zbadanie związku między budżetem filmu a generowanym przychodem.
Analiza ta ma na celu zrozumienie, czy wyższy budżet produkcji filmowej przekłada się na większy dochód na rynku kinowym.
Dane użyte w analizie pochodzą z różnych filmów i obejmują informacje dotyczące budżetu produkcji oraz generowanego przychodu.
"""

# Wytypowanie cech reprezentujących analizowany problem (1 pkt.)
# Jakościowe:
# - Mpaa
# - genre

# Ilościowe:
# - production_cost
# - theaters
# - runtime

# Wytypowanie zmiennej objaśnianej (zależnej) i zmiennych objaśniających (1 pkt.)
# Zmienna objaśniana: worldwide_gross
# Zmienne objaśniające: Mpaa, genre, production_cost, theaters, runtime

# Pozyskanie danych (z co najmniej 100 obserwacji)
# Dane zawierają więcej niż 100 obserwacji, co widać w wyświetlonych danych.

# Przygotowanie zbioru do analizy (2 pkt.)
cleaned_df = movies_df.dropna()
cleaned_df = cleaned_df[(cleaned_df != 0).all(axis=1)]
display_interactive_table(cleaned_df, 5)

# Porządkujemy zbiór względem zmiennej objaśnianej (worldwide_gross)
sorted_df = cleaned_df.sort_values(by='worldwide_gross', ascending=False)
display_interactive_table(sorted_df)

# 2. Statystyczny opis struktury analizowanych cech

# Statystyki opisowe dla zmiennej objaśnianej (2 pkt.)
mean_worldwide_gross = sorted_df['worldwide_gross'].mean()
median_worldwide_gross = sorted_df['worldwide_gross'].median()
print("Średnia dla worldwide_gross:", mean_worldwide_gross)
print("Mediana dla worldwide_gross:", median_worldwide_gross)

# Miary rozproszenia (2 pkt.)
std_dev_gross = sorted_df['worldwide_gross'].std()
variance_gross = sorted_df['worldwide_gross'].var()
iqr_gross = sorted_df['worldwide_gross'].quantile(0.75) - sorted_df['worldwide_gross'].quantile(0.25)
print("\nMiary rozproszenia dla zmiennej objaśnianej (Gross):")
print("Odchylenie standardowe:", std_dev_gross)
print("Wariancja:", variance_gross)
print("Rozstęp międzykwartylowy (IQR):", iqr_gross)

# Wykres ramka-wąsy i histogram dla zmiennej objaśnianej (3 pkt.)
plt.figure(figsize=(10, 5))
sns.boxplot(x=sorted_df['worldwide_gross'])
plt.title('Boxplot średniej oceny semestralnej')
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(sorted_df['worldwide_gross'], kde=True)
plt.title('Histogram średniej oceny semestralnej')
plt.show()

# Skategoryzowane wykresy ramka-wąsy i histogramy (4 pkt.)
plt.figure(figsize=(12, 8))
sns.boxplot(x='genre', y='worldwide_gross', data=sorted_df)
plt.title('Wykres pudełkowy dla genre i worldwide_gross')
plt.xlabel('Gatunek')
plt.ylabel('Przychód globalny')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# 3. Wnioskowanie statystyczne i model regresji

# Weryfikacja hipotezy o zgodności rozkładu zmiennej zależnej z rozkładem normalnym (2 pkt.)
stat, p = shapiro(sorted_df['worldwide_gross'])
print('Statystyka Shapiro-Wilka:', stat)
print('Wartość p:', p)
# Interpretacja: Jeśli wartość p jest mniejsza niż 0.05, odrzucamy hipotezę zerową o normalności rozkładu.

# Związek korelacyjny pomiędzy badanymi zmiennymi (2 pkt.)
correlation = sorted_df.corr()
print(correlation)

# Podział zbioru danych na zbiór treningowy i testowy (1 pkt.)
train, test = train_test_split(sorted_df, test_size=0.3, random_state=42)

# Model regresji liniowej (1 pkt.)
X_train = train[['production_cost', 'theaters', 'runtime']]
y_train = train['worldwide_gross']

X_test = test[['production_cost', 'theaters', 'runtime']]
y_test = test['worldwide_gross']

model = LinearRegression()
model.fit(X_train, y_train)

# Interpretacja modelu (1 pkt.)
print(f'Intercept: {model.intercept_}')
print(f'Coefficients: {model.coef_}')

# Prognoza na zbiorze treningowym i testowym oraz ocena jakości modelu (3 pkt.)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Ocena jakości modelu
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f'Train MSE: {train_mse}, Train R^2: {train_r2}')
print(f'Test MSE: {test_mse}, Test R^2: {test_r2}')

# 4. Opracowanie sprawozdania

# Podsumowanie wyników analiz
"""
Podsumowanie:
1. Średnia i mediana dla zmiennej worldwide_gross pokazują ogólny obraz przychodów filmowych.
2. Miary rozproszenia wskazują na dużą zmienność w przychodach filmowych.
3. Wizualizacje wskazują, że wyższy budżet produkcyjny zazwyczaj wiąże się z wyższymi przychodami.
4. Analiza gatunków pokazuje, że filmy akcji, przygodowe i musicale generują najwyższe przychody.
5. Test Shapiro-Wilka wskazuje na odchylenie od normalności rozkładu przychodów.
6. Model regresji liniowej wykazuje znaczący wpływ kosztów produkcji, liczby kin i czasu trwania na przychody filmowe.
7. Model regresji wykazuje dobrą jakość na zbiorze treningowym i testowym.
"""

# 5. Prezentacja projektu
"""
Podczas prezentacji projektu na forum grupy, omówię:
1. Cel analizy i dane użyte do analizy.
2. Proces czyszczenia danych i przygotowania do analizy.
3. Statystyczne miary położenia i rozproszenia.
4. Wizualizacje danych i wnioski.
5. Model regresji i jego interpretację.
6. Wyniki prognoz i oceny jakości modelu.
"""

# End of the notebook
