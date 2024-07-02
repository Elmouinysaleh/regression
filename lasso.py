# Importation des bibliothèques
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Génération de données synthétiques
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Entraînement du modèle de régression Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = lasso_reg.predict(X_test)

# Évaluation du modèle
mse = mean_squared_error(y_test, y_pred)
print("MSE Régression Lasso:", mse)

# Visualisation des résultats
plt.scatter(X_test, y_test, color='red', label='Données réelles')
plt.plot(X_test, y_pred, color='blue', label='Prédictions')
plt.title('Régression Lasso')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
