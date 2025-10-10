import optuna
import lightgbm as lgb
import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
from config import *
from gain_function import calcular_ganancia, ganancia_lgb_binary, ganancia_evaluator
import matplotlib.pyplot as plt
import seaborn as sns
from grafico_test import crear_grafico_ganancia_avanzado 

def evaluar_en_test(df, mejores_params) -> dict:
    """
    Evalúa el modelo con los mejores hiperparámetros en el conjunto de test.
    Solo calcula la ganancia, sin usar sklearn.
  
    Args:
        df: DataFrame con todos los datos
        mejores_params: Mejores hiperparámetros encontrados por Optuna
  
    Returns:
        dict: Resultados de la evaluación en test (ganancia + estadísticas básicas)
    """
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    logger.info(f"Período de test: {MES_TEST}")
  
    # Preparar datos de entrenamiento (TRAIN + VALIDACION)
    if isinstance(MES_TRAIN, list):
        periodos_entrenamiento = MES_TRAIN + [MES_VALIDACION]
    else:
        periodos_entrenamiento = [MES_TRAIN, MES_VALIDACION]
  
    df_train_completo = df[df['foto_mes'].isin(periodos_entrenamiento)]

    df_test = df[df['foto_mes'] == MES_TEST]
    y_test = df_test['target']
    X_test = df_test.drop(columns=['target'])
    X_train = df_train_completo.drop(columns=['target'])
    y_train = df_train_completo['target']

    # Defino el modelo con los mejores hiperparametros para evaluar en test
    params = mejores_params.copy()
    params.update({
        'objective': 'binary',
        'metric': 'custom',
        'boosting_type': 'gbdt',
        'first_metric_only': True,
        'boost_from_average': True,
        'feature_pre_filter': False,
        'max_bin': 31,
        'seed': SEMILLA[0],
        'verbose': -1
    })

    lgb_train = lgb.Dataset(X_train, label=y_train)

    gbm = lgb.train(
        params,
        lgb_train,
        feval=ganancia_evaluator,
        callbacks=[
            lgb.log_evaluation(period=50)
        ],
    )

    # Predecir en conjunto de test
    y_pred_proba = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    y_pred_binary = (y_pred_proba > UMBRAL).astype(int)

    # Calcular solo la ganancia
    ganancia_test = ganancia_evaluator(y_test, y_pred_binary)
  
    # Estadísticas básicas
    total_predicciones = len(y_pred_binary)
    predicciones_positivas = np.sum(y_pred_binary == 1)
    porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100
  
    resultados = {
        'ganancia_test': float(ganancia_test),
        'total_predicciones': int(total_predicciones),
        'predicciones_positivas': int(predicciones_positivas),
        'porcentaje_positivas': float(porcentaje_positivas)
    }
  
    return resultados, y_pred_proba, y_test

def guardar_resultados_test(resultados_test, archivo_base=None):
    """
    Guarda los resultados de la evaluación en test en un archivo JSON.
    """
    # Guarda en resultados/{STUDY_NAME}_test_results.json
    # ... Implementar utilizando la misma logica que cuando guardamos una iteracion de la Bayesian Optimization
    if archivo_base is None:
        archivo_base = STUDY_NAME
    
    archivo = f"resultados/{archivo_base}_test_results.json"
    with open(archivo, 'w') as f:
        json.dump(resultados_test, f, indent=2)
    logger.info(f"Resultados de test guardados en {archivo}")


def muestrear_ganancias(y_true, y_pred_proba, n_muestras=1000, tamaño_muestra=0.5):
    """
    Realiza muestreos aleatorios sobre los datos de test para estimar la distribución
    de ganancias esperadas.

    Parameters
    ----------
    y_true : array-like
        Valores reales del target (0 o 1).
    y_pred_proba : array-like
        Probabilidades predichas por el modelo.
    n_muestras : int
        Número de simulaciones (default=1000)
    tamaño_muestra : float
        Proporción del dataset usada en cada simulación (default=0.5)

    Returns
    -------
    np.ndarray : vector con ganancias simuladas
    """
    n = len(y_true)
    tamaño = int(n * tamaño_muestra)
    ganancias = []

    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred_proba})
    df = df.sort_values("y_pred", ascending=False)

    for _ in range(n_muestras):
        sample = df.sample(n=tamaño, replace=False)
        # Ganancia simple (puedes cambiar por tu función de ganancia real)
        gan = ganancia_evaluator(sample["y_pred"], sample["y_true"])
        ganancias.append(gan)

    return np.array(ganancias)



def graficar_distribucion_ganancia(ganancias, modelo_nombre, output_dir="resultados/plots"):
    """
    Genera y guarda un histograma + KDE de las ganancias simuladas.

    Parameters
    ----------
    ganancias : np.ndarray
        Ganancias simuladas
    modelo_nombre : str
        Nombre del modelo (para título y nombre del archivo)
    output_dir : str
        Carpeta donde se guardará la imagen
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.histplot(ganancias, kde=True, bins=30, color="steelblue", alpha=0.7)
    plt.axvline(np.mean(ganancias), color="red", linestyle="--", label="Media")
    plt.title(f"Distribución de Ganancia - {modelo_nombre}")
    plt.xlabel("Ganancia simulada")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.tight_layout()

    path_salida = os.path.join(output_dir, f"ganancia_{modelo_nombre}.png")
    plt.savefig(path_salida, dpi=150)
    plt.close()


def registrar_resultados_modelo(modelo_nombre, ganancias, csv_path="resultados/curvas_modelos.csv"):
    """
    Guarda estadísticas de la distribución de ganancias de un modelo
    en un CSV acumulativo (uno por modelo).

    Parameters
    ----------
    modelo_nombre : str
        Nombre del modelo
    ganancias : np.ndarray
        Vector con ganancias simuladas
    csv_path : str
        Ruta al archivo CSV donde se acumulan los resultados
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    resumen = {
        "modelo": modelo_nombre,
        "ganancia_media": np.mean(ganancias),
        "ganancia_std": np.std(ganancias),
        "ganancia_p5": np.percentile(ganancias, 5),
        "ganancia_p95": np.percentile(ganancias, 95),
        "fecha": pd.Timestamp.now()
    }

    df_row = pd.DataFrame([resumen])

    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df_new = pd.concat([df_old, df_row], ignore_index=True)
    else:
        df_new = df_row

    df_new.to_csv(csv_path, index=False)



# def evaluar_en_test_v2(df, mejores_params) -> dict:
#     """
#     Evalúa el modelo con los mejores hiperparámetros en el conjunto de test,
#     entrenando con todas las semillas definidas en config.py y generando
#     el gráfico de ganancia avanzada promedio.

#     Args:
#         df: DataFrame con todos los datos.
#         mejores_params: Mejores hiperparámetros encontrados por Optuna.

#     Returns:
#         dict: Resultados consolidados (media y desvío de la ganancia, etc.)
#     """
#     logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
#     logger.info(f"Período de test: {MES_TEST}")
#     logger.info(f"Usando semillas: {SEMILLA}")

#     # Preparar datos
#     if isinstance(MES_TRAIN, list):
#         periodos_entrenamiento = MES_TRAIN + [MES_VALIDACION]
#     else:
#         periodos_entrenamiento = [MES_TRAIN, MES_VALIDACION]

#     df_train_completo = df[df['foto_mes'].isin(periodos_entrenamiento)]
#     df_test = df[df['foto_mes'] == MES_TEST]

#     X_train = df_train_completo.drop(columns=['target'])
#     y_train = df_train_completo['target']
#     X_test = df_test.drop(columns=['target'])
#     y_test = df_test['target']

#     # Para guardar resultados de todas las semillas
#     resultados_semillas = []
#     predicciones_acumuladas = np.zeros(len(X_test))

#     for i, seed in enumerate(SEMILLA):
#         logger.info(f"Entrenando modelo con semilla {seed} ({i+1}/{len(SEMILLA)})")

#         params = mejores_params.copy()
#         params.update({
#             'objective': 'binary',
#             'metric': 'custom',
#             'boosting_type': 'gbdt',
#             'first_metric_only': True,
#             'boost_from_average': True,
#             'feature_pre_filter': False,
#             'max_bin': 31,
#             'seed': seed,
#             'verbose': -1
#         })

#         lgb_train = lgb.Dataset(X_train, label=y_train)

#         gbm = lgb.train(
#             params,
#             lgb_train,
#             feval=ganancia_evaluator,
#             callbacks=[lgb.log_evaluation(period=100)],
#         )

#         # Predicción con esta semilla
#         y_pred_proba = gbm.predict(X_test, num_iteration=gbm.best_iteration)
#         predicciones_acumuladas += y_pred_proba / len(SEMILLA)  # promedio de predicciones

#         # Ganancia binaria individual
#         y_pred_binaria = (y_pred_proba > UMBRAL).astype(int)
#         ganancia = calcular_ganancia(y_test, y_pred_binaria)
#         resultados_semillas.append(ganancia)
#         logger.info(f"Ganancia con semilla {seed}: {ganancia:,.2f}")

#     # Promedio de ganancias
#     ganancia_media = np.mean(resultados_semillas)
#     ganancia_std = np.std(resultados_semillas)

#     logger.info(f"Ganancia media (test): {ganancia_media:,.2f} ± {ganancia_std:,.2f}")

#     # Generar gráfico avanzado usando el promedio de predicciones
#     titulo = f"Ganancia promedio ({len(SEMILLA)} semillas)"
#     ruta_ganancia = crear_grafico_ganancia_avanzado(y_test, predicciones_acumuladas, titulo)

#     resultados = {
#         'ganancia_media': float(ganancia_media),
#         'ganancia_std': float(ganancia_std),
#         'ganancias_por_semilla': [float(g) for g in resultados_semillas],
#         'grafico_ganancia': ruta_ganancia
#     }

#     return resultados


def evaluar_en_test_v2(df, mejores_params) -> dict:
    """
    Evalúa el modelo con los mejores hiperparámetros en el conjunto de test,
    entrenando con todas las semillas definidas en config.py y generando
    el gráfico de ganancia avanzada de cada semilla.

    Args:
        df: DataFrame con todos los datos.
        mejores_params: Mejores hiperparámetros encontrados por Optuna.

    Returns:
        dict: Resultados consolidados (media y desvío de la ganancia, etc.)
    """
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    logger.info(f"Período de test: {MES_TEST}")
    logger.info(f"Usando semillas: {SEMILLA}")

    # --- Preparar datos ---
    if isinstance(MES_TRAIN, list):
        periodos_entrenamiento = MES_TRAIN + [MES_VALIDACION]
    else:
        periodos_entrenamiento = [MES_TRAIN, MES_VALIDACION]

    df_train_completo = df[df['foto_mes'].isin(periodos_entrenamiento)]
    df_test = df[df['foto_mes'] == MES_TEST]

    X_train = df_train_completo.drop(columns=['target'])
    y_train = df_train_completo['target']
    X_test = df_test.drop(columns=['target'])
    y_test = df_test['target']

    # --- Variables para resultados ---
    resultados_semillas = []
    predicciones_semillas = []

    for i, seed in enumerate(SEMILLA):
        logger.info(f"Entrenando modelo con semilla {seed} ({i+1}/{len(SEMILLA)})")

        params = mejores_params.copy()
        params.update({
            'objective': 'binary',
            'metric': 'custom',
            'boosting_type': 'gbdt',
            'first_metric_only': True,
            'boost_from_average': True,
            'feature_pre_filter': False,
            'max_bin': 31,
            'seed': seed,
            'verbose': -1,
            'extra_trees': False  # Para mayor diversidad entre semillas
        })

        lgb_train = lgb.Dataset(X_train, label=y_train)

        gbm = lgb.train(
            params,
            lgb_train,
            feval=ganancia_evaluator,
            callbacks=[lgb.log_evaluation(period=100)],
        )

        # Predicción con esta semilla
        y_pred_proba = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        predicciones_semillas.append(y_pred_proba)

        # Ganancia binaria individual
        y_pred_binaria = (y_pred_proba > UMBRAL).astype(int)
        ganancia = calcular_ganancia(y_test, y_pred_binaria)
        resultados_semillas.append(ganancia)
        logger.info(f"Ganancia con semilla {seed}: {ganancia:,.2f}")

    # --- Promedio de ganancias ---
    ganancia_media = np.mean(resultados_semillas)
    ganancia_std = np.std(resultados_semillas)
    logger.info(f"Ganancia media (test): {ganancia_media:,.2f} ± {ganancia_std:,.2f}")

    # --- Gráfico con todas las semillas ---
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for i, (seed, y_pred_proba) in enumerate(zip(SEMILLA, predicciones_semillas)):
        # La función crear_grafico_ganancia_avanzado devuelve (xs, ys) si la adaptás
        xs, ys = obtener_curva_ganancia(y_test, y_pred_proba)  # función auxiliar
        plt.plot(xs, ys, label=f"Semilla {seed}", alpha=0.8)

    plt.title(f"Curvas de ganancia por semilla ({len(SEMILLA)} semillas)")
    plt.xlabel("Cantidad de clientes contactados")
    plt.ylabel("Ganancia acumulada")
    plt.legend()
    plt.grid(True)
    ruta_grafico = f"resultados/plots/ganancia_semillas_{STUDY_NAME}.png"
    plt.savefig(ruta_grafico, bbox_inches="tight")
    plt.close()

    resultados = {
        'ganancia_media': float(ganancia_media),
        'ganancia_std': float(ganancia_std),
        'ganancias_por_semilla': [float(g) for g in resultados_semillas],
        'grafico_ganancia': ruta_grafico
    }

    return resultados



def obtener_curva_ganancia(y_true, y_pred_proba):
    """
    Calcula la curva de ganancia acumulada a partir de predicciones probabilísticas.
    
    Args:
        y_true: array-like, valores reales (0/1)
        y_pred_proba: array-like, probabilidades predichas
    
    Returns:
        xs: número acumulado de clientes contactados (ordenados por probabilidad descendente)
        ys: ganancia acumulada correspondiente
    """
    import numpy as np

    # Ordenar por probabilidad descendente
    orden = np.argsort(-y_pred_proba)
    y_true_sorted = y_true.iloc[orden] if hasattr(y_true, "iloc") else y_true[orden]

    # Ganancia acumulada: 1 cliente bueno = +$1, 1 cliente malo = -$10 (ejemplo, adaptá a tu cálculo real)
    # ganancia_unitaria = np.where(y_true_sorted == 1, 1, -10)
    # ganancia_acumulada = np.cumsum(ganancia_unitaria)
    ganancia_acumulada = ganancia_evaluator(y_true_sorted, np.ones_like(y_true_sorted))

    # Eje X: número de clientes contactados
    xs = np.arange(1, len(y_true_sorted) + 1)
    ys = ganancia_acumulada

    return xs, ys
