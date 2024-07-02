# Importation des bibliothèques
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Génération de données synthétiques
np.random.seed(0)
X = 2 * np.random.rand(100, 2)
y = 4 + 3 * X[:, 0] + 5 * X[:, 1] + np.random.randn(100)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Entraînement du modèle de régression linéaire multiple
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = lin_reg.predict(X_test)

# Évaluation du modèle
mse = mean_squared_error(y_test, y_pred)
print("MSE Régression Linéaire Multiple:", mse)

# Visualisation des résultats
plt.scatter(y_test, y_pred, color='blue', label='Prédictions vs Réelles')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.title('Régression Linéaire Multiple')
plt.xlabel('Valeurs réelles')
plt.ylabel('Valeurs prédites')
plt.legend()
plt.show()
