# Importation des bibliothèques
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Génération de données synthétiques
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + X**2 + np.random.randn(100, 1)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Transformation des caractéristiques pour la régression polynomiale
degree = 2
poly_features = PolynomialFeatures(degree=degree)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# Entraînement du modèle de régression polynomiale
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)

# Prédiction sur l'ensemble de test
y_pred = poly_reg.predict(X_test_poly)

# Évaluation du modèle
mse = mean_squared_error(y_test, y_pred)
print(f"MSE Régression Polynomiale (degré {degree}):", mse)

# Visualisation des résultats
plt.scatter(X_test, y_test, color='red', label='Données réelles')
plt.plot(np.sort(X_test, axis=0), np.sort(y_pred, axis=0), color='blue', label='Prédictions')
plt.title(f'Régression Polynomiale (degré {degree})')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
