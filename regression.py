import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import sys
import os

def load_data_from_csv(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"Erreur : Le fichier {file_path} n'a pas été trouvé.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Erreur : Le fichier {file_path} est vide.")
        return None
    except pd.errors.ParserError:
        print(f"Erreur : Le fichier {file_path} contient des erreurs de syntaxe.")
        return None

def generate_data(regression_type):
    np.random.seed(0)
    if regression_type == "simple":
        X = 2 * np.random.rand(100, 1)
        y = 4 + 3 * X + np.random.randn(100, 1)
    elif regression_type == "multiple":
        X = 2 * np.random.rand(100, 2)
        y = 4 + 3 * X[:, 0] + 5 * X[:, 1] + np.random.randn(100)
    elif regression_type == "logistic":
        X = np.random.rand(100, 1) * 10
        y = (X[:, 0] > 5).astype(int)
    elif regression_type == "polynomial":
        X = 6 * np.random.rand(100, 1) - 3
        y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1)
    elif regression_type in ["ridge", "lasso"]:
        X = 2 * np.random.rand(100, 1)
        y = 4 + 3 * X + np.random.randn(100, 1)
    return X, y

def simple_linear_regression(X=None, y=None):
    if X is None or y is None:
        X, y = generate_data("simple")
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    y_pred = lin_reg.predict(X)
    print(f"Coefficients: {lin_reg.coef_}, Intercept: {lin_reg.intercept_}")
    plt.scatter(X, y, color='blue')
    plt.plot(X, y_pred, color='red')
    plt.title('Régression Linéaire Simple')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()

def multiple_linear_regression(X=None, y=None):
    if X is None or y is None:
        X, y = generate_data("multiple")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred = lin_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Erreur quadratique moyenne: {mse}")
    print(f"Coefficients: {lin_reg.coef_}, Intercept: {lin_reg.intercept_}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_test[:, 0], X_test[:, 1], y_test, color='blue', label='Données réelles')
    ax.scatter(X_test[:, 0], X_test[:, 1], y_pred, color='red', label='Prédictions')
    ax.set_title('Régression Linéaire Multiple')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('y')
    plt.legend()
    plt.show()

def logistic_regression(X=None, y=None):
    if X is None or y is None:
        X, y = generate_data("logistic")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Précision: {accuracy}")
    plt.scatter(X_test, y_test, color='blue', label='Données réelles')
    plt.plot(X_test, log_reg.predict_proba(X_test)[:, 1], color='red', label='Probabilité prédite')
    plt.title('Régression Logistique')
    plt.xlabel('X')
    plt.ylabel('Probabilité')
    plt.legend()
    plt.show()

def polynomial_regression(X=None, y=None):
    if X is None or y is None:
        X, y = generate_data("polynomial")
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly, y)
    y_pred = poly_reg.predict(X_poly)
    print(f"Coefficients: {poly_reg.coef_}, Intercept: {poly_reg.intercept_}")
    plt.scatter(X, y, color='blue')
    plt.plot(np.sort(X, axis=0), y_pred[np.argsort(X[:, 0])], color='red')
    plt.title('Régression Polynomiale')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()

def ridge_regression(X=None, y=None):
    if X is None or y is None:
        X, y = generate_data("ridge")
    ridge_reg = Ridge(alpha=1, solver="cholesky")
    ridge_reg.fit(X, y)
    y_pred = ridge_reg.predict(X)
    print(f"Coefficients: {ridge_reg.coef_}, Intercept: {ridge_reg.intercept_}")
    plt.scatter(X, y, color='blue')
    plt.plot(X, y_pred, color='red')
    plt.title('Régression Ridge')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()

def lasso_regression(X=None, y=None):
    if X is None or y is None:
        X, y = generate_data("lasso")
    lasso_reg = Lasso(alpha=0.1)
    lasso_reg.fit(X, y)
    y_pred = lasso_reg.predict(X)
    print(f"Coefficients: {lasso_reg.coef_}, Intercept: {lasso_reg.intercept_}")
    plt.scatter(X, y, color='blue')
    plt.plot(X, y_pred, color='red')
    plt.title('Régression Lasso')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()

def create_csv_template():
    template = pd.DataFrame({
        "X1": [],
        "X2": [],
        "y": []
    })
    template_path = "template.csv"
    if os.path.exists(template_path):
        print(f"Avertissement : Le fichier {template_path} existe déjà et sera écrasé.")
        try:
            os.remove(template_path)
        except PermissionError:
            print(f"Erreur : Impossible de supprimer le fichier {template_path} en raison de permissions insuffisantes.")
            return
    try:
        template.to_csv(template_path, index=False)
        print(f"Template CSV créé : {os.path.abspath(template_path)}")
    except PermissionError:
        print(f"Erreur : Impossible de créer le fichier {template_path} en raison de permissions insuffisantes.")

def main():
    while True:
        print("\nMenu:")
        print("1. Régression Linéaire Simple")
        print("2. Régression Linéaire Multiple")
        print("3. Régression Logistique")
        print("4. Régression Polynomiale")
        print("5. Régression Ridge")
        print("6. Régression Lasso")
        print("7. Charger un dataset via un fichier CSV")
        print("8. Télécharger le modèle CSV")
        print("0. Quitter")
        choice = input("Entrez votre choix: ")

        if choice == '1':
            simple_linear_regression()
        elif choice == '2':
            multiple_linear_regression()
        elif choice == '3':
            logistic_regression()
        elif choice == '4':
            polynomial_regression()
        elif choice == '5':
            ridge_regression()
        elif choice == '6':
            lasso_regression()
        elif choice == '7':
            file_path = input("Entrez le chemin du fichier CSV: ")
            data = load_data_from_csv(file_path)
            if data is not None:
                try:
                    print("\nDonnées chargées avec succès. Veuillez choisir le type de régression à appliquer.")
                    print("Colonnes dans le dataset: ", data.columns)
                    x_columns = input("Entrez les noms des colonnes des prédicteurs (séparés par des virgules): ").split(',')
                    y_column = input("Entrez le nom de la colonne cible: ")

                    # Vérifier si les colonnes spécifiées existent dans le dataset
                    if not all(col in data.columns for col in x_columns):
                        raise KeyError(f"Les colonnes spécifiées {x_columns} ne se trouvent pas dans le dataset.")
                    if y_column not in data.columns:
                        raise KeyError(f"La colonne cible spécifiée {y_column} ne se trouve pas dans le dataset.")

                    X = data[x_columns].values
                    y = data[y_column].values
                    regression_choice = input("\nChoisissez le type de régression à appliquer:\n1. Régression Linéaire Simple\n2. Régression Linéaire Multiple\n3. Régression Logistique\n4. Régression Polynomiale\n5. Régression Ridge\n6. Régression Lasso\nEntrez votre choix: ")
                    if regression_choice == '1':
                        simple_linear_regression(X, y)
                    elif regression_choice == '2':
                        multiple_linear_regression(X, y)
                    elif regression_choice == '3':
                        logistic_regression(X, y)
                    elif regression_choice == '4':
                        polynomial_regression(X, y)
                    elif regression_choice == '5':
                        ridge_regression(X, y)
                    elif regression_choice == '6':
                        lasso_regression(X, y)
                    else:
                        print("Choix invalide. Veuillez réessayer.")
                except KeyError as e:
                    print(f"Erreur : {e}. Veuillez vérifier les noms des colonnes spécifiées.")
            else:
                print("Échec du chargement des données.")
        elif choice == '8':
            create_csv_template()
        elif choice == '0':
            sys.exit()
        else:
            print("Choix invalide. Veuillez réessayer.")

if __name__ == "__main__":
    main()
