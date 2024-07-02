# Importation des bibliothèques
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Génération de données synthétiques
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = (X > 1).astype(int).ravel()

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Entraînement du modèle de régression logistique
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = log_reg.predict(X_test)

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Régression Logistique:", accuracy)

# Visualisation des résultats
plt.scatter(X_test, y_test, color='red', label='Données réelles')
plt.plot(X_test, log_reg.predict_proba(X_test)[:, 1], color='blue', label='Probabilités prédites')
plt.title('Régression Logistique')
plt.xlabel('X')
plt.ylabel('Probabilité')
plt.legend()
plt.show()
