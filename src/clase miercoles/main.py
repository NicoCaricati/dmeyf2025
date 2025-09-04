# No hacer scripts sueltos de cada tema. Ir integrando todo en un solo script.
# Dejar todo en un archivo bastante limpio y ordenado.
# Hasta ahora agregar clase ternaria, arboles, bosques, obstuza
import pandas as pd
import numpy as np
import os
import optuna
import sklearn


def main():
    print("Inicio de Ejecucion")

    # Carga de datos
    try:
        df = pd.read_csv("data/competencia_01.csv")
        print(df.head())
        print(df.columns)
        print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
        print("Dataset Cargado")
    except FileNotFoundError:
        print("Error: El archivo 'competencia_01.csv' no se encuentra.")
        return
    
    # Semilla para reproducibilidad
    semillas = [555557, 555871, 556219, 556459, 556691]

    # Funcion de Ganancia
    ganancia_acierto = 780000
    costo_estimulo = 20000

    def ganancia(model, X, y, prop=1, threshold=0.025):

        class_index = np.where(model.classes_ == "BAJA+2")[0][0]
        y_hat = model.predict_proba(X)

        @np.vectorize
        def ganancia_row(predicted, actual, threshold=0.025):
            return  (predicted >= threshold) * (ganancia_acierto if actual == "BAJA+2" else -costo_estimulo)

        return ganancia_row(y_hat[:,class_index], y).sum() / prop
    
    # Preprocesamiento?
    df_train = df[df["foto_mes"] <= 202104]
    X = df_train.drop(columns=["target"])
    y = df_train["target"]

    # Bosques no soportan NAs, ver que variables tienen NAs y como imputarlas
    print(X.isna().sum())



    # Separacion de datos

    # Tres metodos

    # Train, test, validation
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.7, random_state=semillas[0])
    X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=semillas[0])

    # Cross validation (todo de Copilot, chequear)

    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=semillas[0])
    for train_index, test_index in skf.split(X, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Shuffle Split

    sss = StratifiedShuffleSplit(n_splits=5,
                             test_size=0.3,
                             random_state=semillas[0])
    
    for train_index, test_index in sss.split(X, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]



    # Optimizacion de hiperparametros




if __name__ == "__main__":
    main()
    


print("Hola Mundo")