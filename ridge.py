# Importation des bibliothèques
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Génération de données synthétiques
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Entraînement du modèle de régression Ridge
ridge_reg = Ridge(alpha=1)
ridge_reg.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = ridge_reg.predict(X_test)

# Évaluation du modèle
mse = mean_squared_error(y_test, y_pred)
print("MSE Régression Ridge:", mse)

# Visualisation des résultats
plt.scatter(X_test, y_test, color='red', label='Données réelles')
plt.plot(X_test, y_pred, color='blue', label='Prédictions')
plt.title('Régression Ridge')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
