# La idea del entorno virtual es SIMULAR lo q vamos a tener q hacer en un servidor.
# Dejar en claro en requirements.txt las librerias que vamos a usar, y sus versiones.

# No hacer scripts sueltos de cada tema. Ir integrando todo en un solo script.
# Dejar todo en un archivo bastante limpio y ordenado.
# Hasta ahora agregar clase ternaria, arboles, bosques, obstuza

# Jupyter notebook sirve para EDA, visualizaciones
# Py files para crear variables, preprocesamiento ORDENADO, hiperparametros, modelos, predicciones

import pandas as pd
import numpy as np
import os
import optuna
import sklearn

def main():
    print("Inicio de Ejecucion")

    # Carga de datos
    try:
        df = pd.read_csv("data/competencia_03.csv")
        print(df.head())
        print(df.columns)
        print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
        print("Dataset Cargado")
    except FileNotFoundError:
        print("Error: El archivo 'competencia_03.csv' no se encuentra.")
        return
    
    with open("logs/logs.txt", "a") as f:
        f.write(f"Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas\n")
    
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
    print("Iniciando preprocesamiento")
    # Bosques no soportan NAs, ver que variables tienen NAs y como imputarlas
    # print(X.isna().sum())

    # Preprocesamiento a df, luego separo df_train y df_to_predict

    df_train = df[df["foto_mes"] <= 202104]
    X = df_train.drop(columns=["target"])
    y = df_train["target"]

    with open("logs/logs.txt", "a") as f:
        f.write(f"Preprocesamiento finalizado \n")

    # Separacion de datos
    print("Iniciando separacion de datos")
    # Tres metodos

    # # Train, test, validation
    # from sklearn.model_selection import train_test_split
    # X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.7, random_state=semillas[0])
    # X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=semillas[0])

    # Cross validation (todo de Copilot, chequear)

    # from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=semillas[0])
    # for train_index, test_index in skf.split(X, y):
    #     print("TRAIN:", train_index, "TEST:", test_index)
    #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    #     y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # # # Shuffle Split
    # print("Iniciando Shuffle Split")
    # from sklearn.model_selection import StratifiedShuffleSplit
    # sss = StratifiedShuffleSplit(n_splits=5,
    #                          test_size=0.3,
    #                          random_state=semillas[0])

    # # Aplico shuffle split
    # sss.split(X, y)
    # print(sss.get_n_splits(X, y))

    # with open("logs/logs.txt", "a") as f:
    #     f.write(f"Particion finalizada \n")   


    
#     # for train_index, test_index in sss.split(X, y):
#     #     print("TRAIN:", train_index, "TEST:", test_index)
#     #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     #     y_train, y_test = y.iloc[train_index], y.iloc[test_index]



#     # Optimizacion de hiperparametros con Optuna para Random Forest
    """     print("Iniciando Optimizacion de Hiperparametros")
    from sklearn.tree import DecisionTreeClassifier 
    from joblib import Parallel, delayed
    criterion = 'gini'
    def train_and_evaluate(train_index, test_index, X, y, params):
        m = DecisionTreeClassifier(**params, random_state=semillas[0])
        m.fit(X.iloc[train_index], y.iloc[train_index])
        return ganancia(m, X.iloc[test_index], y.iloc[test_index], prop=0.3)

    def objective(trial, X, y, sss):
        params = {
            "criterion": "gini",
            "max_depth": trial.suggest_int("max_depth", 6, 24),
            "min_samples_split": trial.suggest_int("min_samples_split", 100, 10000),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 50, 5000),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 64, 4096),
        }

        results = Parallel(n_jobs=-1)(
            delayed(train_and_evaluate)(train_idx, test_idx, X, y, params)
            for train_idx, test_idx in sss.split(X, y)
        )
        return np.mean(results)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y, sss), n_trials
                     =50, n_jobs=1)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    with open("logs/logs.txt", "a") as f:
        f.write(f"Optimizacion finalizada \n")
        f.write(f"Best trial: {trial.value} \n")
        f.write(f"Params: {trial.params} \n")
    print("Optimizacion finalizada")
     """

    # Prediccion con el mejor modelo
    print("Iniciando Prediccion")
    from sklearn.tree import DecisionTreeClassifier
    from datetime import datetime
    df_to_predict = df[df["foto_mes"] == 202106]
    X_prediction = df_to_predict.drop(columns=["target"])
    Params = {'max_depth': 16, 'min_samples_split': 4343, 'min_samples_leaf': 265, 'max_leaf_nodes': 2953} 
    model = DecisionTreeClassifier(criterion='gini',
                               random_state=semillas[0],
                               **Params)
    model.fit(X, y)
    class_index = np.where(model.classes_ == "BAJA+2")[0][0]
    probs = model.predict_proba(X_prediction)[:, class_index]
    threshold = 0.025
    y_pred = (probs >= threshold).astype(int)

    # Archivo para Kaggle
    submission = pd.DataFrame({
    "numero_de_cliente": df_to_predict["numero_de_cliente"],
    "Predicted": y_pred
    })
    hora_actual = datetime.now().strftime("%B %d %H:%M")
    submission.to_csv("predicciones_rf_202106.csv", index=False)
    print(f"CSV generado: predicciones_rf{hora_actual}.csv")
    with open("logs/logs.txt", "a") as f:
            f.write(f"Prediccion finalizada \n")

    print("Fin de Ejecucion")




if __name__ == "__main__":
    main()
    


# Feature Engenineering