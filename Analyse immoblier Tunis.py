import os
os.environ['TCL_LIBRARY'] = r'C:\Users\DELL\AppData\Local\Programs\Python\Python313\tcl\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\Users\DELL\AppData\Local\Programs\Python\Python313\tcl\tk8.6'
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor

# Charger les données depuis un fichier .xlsx
url = r"C:\projet tutoré\IMMOBILIER.xlsx"
data = pd.read_excel(url)

# Aperçu des données
print("Aperçu des données :")
print(data.head())

# Gestion des valeurs manquantes
print("\nValeurs manquantes par colonne :")
print(data.isnull().sum())
data = data.loc[:, data.isnull().mean() < 0.7]

# Remplir les valeurs manquantes
for col in data.select_dtypes(include=['int64', 'float64']):
    data[col] = data[col].fillna(data[col].median())
for col in data.select_dtypes(include=['object']):
    data[col] = data[col].fillna(data[col].mode()[0])

# Suppression des doublons
data = data.drop_duplicates()

# Encodage des variables catégoriques
label_encoder = LabelEncoder()
for col in data.select_dtypes(include=['object']):
    data[col] = label_encoder.fit_transform(data[col])

data['avg_price_by_zone'] = data.groupby('Zone')['price'].transform('mean')
data['avg_price_by_size'] = data.groupby('size')['price'].transform('mean')

# Analyse exploratoire
plt.figure(figsize=(12, 10))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Heatmap des corrélations après transformations")
plt.show()

# Séparation des variables explicatives et de la cible
X = data.drop("price", axis=1)
y = data["price"]

# Sélection avancée des caractéristiques
selector = SelectKBest(score_func=f_regression, k=10)
X = selector.fit_transform(X, y)

# Séparation des ensembles
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèles de base pour le stacking
base_models = [
    ("Random Forest", RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=4, random_state=42)),
    ("XGBoost", XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, subsample=0.7, random_state=42)),
    ("LightGBM", lgb.LGBMRegressor(boosting_type='gbdt', num_leaves=31, learning_rate=0.05, feature_fraction=0.7, random_state=42)),
    ("CatBoost", CatBoostRegressor(iterations=500, learning_rate=0.05, depth=8, verbose=0, random_seed=42))
]


from sklearn.ensemble import GradientBoostingRegressor

stacking_model = StackingRegressor(
    estimators=base_models, final_estimator=GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
)


# Initialisation d'un dictionnaire pour stocker les résultats
results = {}

# Évaluation des modèles
for name, model in base_models:
    print(f"--- Évaluation du modèle : {name} ---")
    model.fit(X_train, y_train)  # Entraînement du modèle
    y_pred = model.predict(X_test)  # Prédictions sur le test

    # Calcul des métriques
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Affichage des résultats
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}\n")

    # Stockage des résultats
    results[name] = {"RMSE": rmse, "R²": r2}

# Affichage des résultats sous forme de tableau
print("\n--- Résumé des performances des modèles ---")
results_df = pd.DataFrame(results).T
print(results_df)

# Validation croisée pour générer des prédictions fiables sur les données d'entraînement
y_train_pred_stack = cross_val_predict(stacking_model, X_train, y_train, cv=5)

# Entraînement final sur l'ensemble des données d'entraînement
stacking_model.fit(X_train, y_train)

# Prédictions sur le test
y_pred_stack = stacking_model.predict(X_test)

# Calcul des métriques
rmse_stack = np.sqrt(mean_squared_error(y_test, y_pred_stack))
r2_stack = r2_score(y_test, y_pred_stack)

# Affichage des résultats
print(f"--- Stacking avec réduction de surapprentissage ---")
print(f"RMSE: {rmse_stack:.2f}")
print(f"R²: {r2_stack:.2f}")

# Évaluation des erreurs résiduelles
residuals = y_test - y_pred_stack
sns.histplot(residuals, kde=True, bins=30)
plt.title("Distribution des erreurs résiduelles (Stacking)")
plt.xlabel("Erreur résiduelle")
plt.ylabel("Fréquence")
plt.show()

# Visualisation des prédictions vs valeurs réelles
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_stack)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
plt.xlabel("Valeurs réelles")
plt.ylabel("Prédictions")
plt.title("Prédictions vs Valeurs réelles (Stacking)")
plt.show()

# Évaluer le modèle de stacking
y_train_pred_stack = stacking_model.predict(X_train)
train_rmse_stack = np.sqrt(mean_squared_error(y_train, y_train_pred_stack))
train_r2_stack = r2_score(y_train, y_train_pred_stack)

y_test_pred_stack = stacking_model.predict(X_test)
test_rmse_stack = np.sqrt(mean_squared_error(y_test, y_test_pred_stack))
test_r2_stack = r2_score(y_test, y_test_pred_stack)

print(f"--- Stacking ---")
print(f"Train RMSE: {train_rmse_stack:.2f}, Train R²: {train_r2_stack:.2f}")
print(f"Test RMSE: {test_rmse_stack:.2f}, Test R²: {test_r2_stack:.2f}")

# Détection des outliers
outliers = data.iloc[np.where(abs(residuals) > 50000)]
print(outliers)

# Validation croisée pour obtenir la RMSE moyenne
scores = cross_val_score(stacking_model, X, y, cv=5, scoring='neg_mean_squared_error')
print(f"Validation RMSE (moyenne) : {np.sqrt(-scores.mean()):.2f}")
