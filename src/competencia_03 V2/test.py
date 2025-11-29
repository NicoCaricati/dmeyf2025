# test
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
import random
from grafico_test import calcular_ganancia_acumulada_optimizada

import lightgbm as lgb
import numpy as np
import pandas as pd
import logging


def evaluar_en_test(df, mejores_params, mes_test, meses_train, seed=SEMILLA[0]) -> dict:
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
    logger.info(f"Período de test: {mes_test}")
  
    # Preparar datos de entrenamiento 
    periodos_entrenamiento = meses_train
  
    df_train_completo = df[df['foto_mes'].isin(periodos_entrenamiento)]

    df_test = df[df['foto_mes'] == mes_test]
    y_test = df_test['target_to_calculate_gan']
    X_test = df_test.drop(columns=['target','target_to_calculate_gan'])
    X_train = df_train_completo.drop(columns=['target', 'target_to_calculate_gan'])
    y_train = df_train_completo['target']

    # def lr_schedule(iteration):
    #     return params_base["lr_init"] * (params_base["lr_decay"] ** iteration)


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
        'seed': seed,
        'verbose': -1
    })

    lgb_train = lgb.Dataset(X_train, label=y_train)

    gbm = lgb.train(
        params,
        lgb_train,
        feval=ganancia_evaluator,
        callbacks=[
            # lgb.reset_parameter(learning_rate=lr_schedule),
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



def evaluar_en_test_ensamble(df, mejores_params, semillas: list[int]) -> dict:
    """
    Evalúa un ensamble de modelos LightGBM entrenados con distintas semillas
    sobre el conjunto de test. Promedia las probabilidades por cliente.

    Args
    ----
    df : pd.DataFrame
        DataFrame con todos los datos.
    mejores_params : dict
        Mejores hiperparámetros encontrados por Optuna.
    semillas : list[int]
        Lista de semillas para entrenar múltiples modelos y promediar.

    Returns
    -------
    tuple[dict, np.ndarray, pd.Series]
        (resultados, y_pred_proba_prom, y_test)
    """
    logger = logging.getLogger(__name__)
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST (ENSEMBLE) ===")
    logger.info(f"Período de test: {MES_TEST}")
    logger.info(f"Semillas utilizadas: {semillas}")

    # Definir períodos de entrenamiento (TRAIN + VALID)
    if isinstance(MES_TRAIN, list):
        periodos_entrenamiento = MES_TRAIN + [MES_VALIDACION]
    else:
        periodos_entrenamiento = [MES_TRAIN, MES_VALIDACION]

    df_train_completo = df[df['foto_mes'].isin(periodos_entrenamiento)]
    df_test = df[df['foto_mes'] == MES_TEST]

    y_test = df_test['target_to_calculate_gan']
    X_test = df_test.drop(columns=['target', 'target_to_calculate_gan'])
    X_train = df_train_completo.drop(columns=['target', 'target_to_calculate_gan'])
    y_train = df_train_completo['target']

    # Acumular predicciones
    predicciones = np.zeros(len(X_test))

    for seed in semillas:
        logger.info(f"Entrenando modelo con semilla {seed}...")
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
            'verbose': -1
        })

        lgb_train = lgb.Dataset(X_train, label=y_train)

        gbm = lgb.train(
            params,
            lgb_train,
            feval=ganancia_evaluator,
            callbacks=[lgb.log_evaluation(period=50)],
        )

        y_pred_proba = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        predicciones += y_pred_proba

    # Promediar predicciones de todas las semillas
    y_pred_proba_prom = predicciones / len(semillas)
    y_pred_binary = (y_pred_proba_prom > UMBRAL).astype(int)

    # Calcular ganancia final
    ganancia_test = ganancia_evaluator(y_test, y_pred_binary)

    # Estadísticas básicas
    total_predicciones = len(y_pred_binary)
    predicciones_positivas = np.sum(y_pred_binary == 1)
    porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100

    resultados = {
        'ganancia_test': float(ganancia_test),
        'total_predicciones': int(total_predicciones),
        'predicciones_positivas': int(predicciones_positivas),
        'porcentaje_positivas': float(porcentaje_positivas),
        'n_modelos_ensemble': len(semillas)
    }

    logger.info(f"✅ Ensamble completado con {len(semillas)} modelos.")
    logger.info(f"Ganancia test: {ganancia_test:,.0f}")
    logger.info(f"Predicciones positivas: {predicciones_positivas} ({porcentaje_positivas:.2f}%)")

    return resultados, y_pred_proba_prom, y_test


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
    
    study_name = STUDY_NAME
    
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    os.makedirs(f"../../../buckets/b1/Compe_02/{study_name}", exist_ok=True)

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
    df_new.to_csv(f"../../../buckets/b1/Compe_02/{study_name}_curvas_modelos.csv", index=False)
    


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


def evaluar_con_varias_semillas(df_fe, mejores_params, semillas, study_name_base="experimento_multi_seed"):
    """
    Evalúa el modelo con distintas semillas, grafica las curvas de ganancia acumulada y 
    muestra la curva promedio.
    """
    logger.info(f"=== Evaluando modelo con {len(semillas)} semillas ===")

    ganancias_por_seed = []
    curvas = []

    for seed in semillas:
        logger.info(f"🔁 Ejecutando con seed = {seed}")

        # Fijar la semilla
        np.random.seed(seed)
        random.seed(seed)

        # Evaluar modelo
        resultados_test, y_pred_proba, y_test = evaluar_en_test(df_fe, mejores_params)

        # Calcular ganancia acumulada
        ganancias_acumuladas, indices_ordenados, _ = calcular_ganancia_acumulada_optimizada(y_test, y_pred_proba)
        curvas.append(ganancias_acumuladas)
        ganancias_por_seed.append(np.max(ganancias_acumuladas))

    # Alinear todas las curvas al mismo largo (por si alguna difiere)
    min_len = min(len(curva) for curva in curvas)
    curvas = [curva[:min_len] for curva in curvas]
    curvas = np.array(curvas)

    # Calcular promedio y desvío
    curva_promedio = np.mean(curvas, axis=0)
    curva_std = np.std(curvas, axis=0)
    
    max_std = np.max(curva_std)
    logger.warning(f"⚠️ MÁXIMA DESVIACIÓN EN LAS CURVAS: {max_std:.6f}")
    if max_std < 0.001:
        logger.warning("Curvas de semillas casi idénticas. La variación es absorbida por el modelo.")
    
    # === Graficar ===
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Graficar cada curva individual
    for i, curva in enumerate(curvas):
        ax.plot(curva, alpha=0.3, lw=1.5, label=f"Seed {semillas[i]}")

    # Graficar curva promedio con intervalo de confianza
    ax.plot(curva_promedio, color='black', lw=3, label='Promedio', zorder=5)
    ax.fill_between(range(min_len),
                    curva_promedio - curva_std,
                    curva_promedio + curva_std,
                    color='gray', alpha=0.2, label='±1σ')

    ax.set_title(f"Ganancia acumulada por semilla - {study_name_base}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Clientes ordenados por probabilidad", fontsize=12)
    ax.set_ylabel("Ganancia acumulada", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    # Guardar resultado
    os.makedirs("resultados/plots", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_grafico = f"resultados/plots/{study_name_base}_ganancia_multi_seed_{timestamp}.png"
    plt.savefig(ruta_grafico, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    logger.info(f"✅ Gráfico multi-seed guardado en: {ruta_grafico}")

    return {
        "ganancias_por_seed": ganancias_por_seed,
        "curva_promedio": curva_promedio,
        "curva_std": curva_std,
        "ruta_grafico": ruta_grafico
    }




def comparar_semillas_en_grafico(df_fe, mejores_params, semillas, study_name="multi_seed"):
    """
    Corre el modelo con distintas semillas y muestra las curvas de ganancia acumulada
    de cada corrida junto con la curva promedio, con ajustes automáticos de ejes
    y marcando el punto óptimo del promedio.
    """
    logger.info(f"=== Comparando {len(semillas)} semillas ===")

    curvas = []
    ganancias_max = []

    # ... (Código para calcular curvas de ganancia acumulada por semilla)
    for seed in semillas:
        logger.info(f"🔁 Semilla: {seed}")
        np.random.seed(seed)
        random.seed(seed)
        resultados_test, y_pred_proba, y_test = evaluar_en_test(df_fe, mejores_params, seed=seed)
        y_test = np.asarray(y_test)
        y_pred_proba = np.asarray(y_pred_proba)
        ganancias_acumuladas, _, _ = calcular_ganancia_acumulada_optimizada(y_test, y_pred_proba)
        curvas.append(ganancias_acumuladas)
        ganancias_max.append(np.max(ganancias_acumuladas))
    # ... (Fin del cálculo de curvas)


    # Emparejar largo de las curvas
    min_len = min(len(c) for c in curvas)
    curvas = np.array([c[:min_len] for c in curvas])

    # Promedio y desviación
    curva_prom = curvas.mean(axis=0)
    curva_std = curvas.std(axis=0)

    # 1. Definir el límite de X (25,000 clientes)
    LIMITE_X = 20000 
    x_max = min(min_len, LIMITE_X)
    
    # ⭐️ CALCULAR PUNTO ÓPTIMO DEL PROMEDIO ⭐️
    curva_prom_cortada = curva_prom[:x_max]
    idx_max = np.argmax(curva_prom_cortada)
    ganancia_optima = curva_prom_cortada[idx_max]
    clientes_optimos = idx_max
    
    # === GRAFICAR ===
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Todas las curvas individuales (Aumentada visibilidad y cortada a x_max)
    for i, curva in enumerate(curvas):
        ax.plot(curva[:x_max], alpha=0.8, lw=1.8, label=f"Seed {semillas[i]}")

    # Curva promedio (Cortada a x_max)
    # Se añade 'zorder=5' para asegurar que la línea promedio esté sobre las individuales
    ax.plot(curva_prom[:x_max], color="black", lw=3, label="Promedio", zorder=5)
    
    # Sombreado
    ax.fill_between(range(x_max), 
                    curva_prom[:x_max] - curva_std[:x_max], 
                    curva_prom[:x_max] + curva_std[:x_max], 
                    color="gray", alpha=0.2)

    # ⭐️ MARCAR EL PUNTO ÓPTIMO EN EL GRÁFICO ⭐️
    # Usamos scatter para un punto, con zorder alto para que sea visible
    ax.scatter(clientes_optimos, ganancia_optima, 
               color='red', 
               s=100, # Tamaño del marcador
               marker='*', # Forma de estrella para destacar
               zorder=10, 
               label=f"Óptimo Promedio: {ganancia_optima:,.0f} ({clientes_optimos} Clientes)")


    # Configuración del gráfico
    ax.set_title(f"Ganancia acumulada por semilla - {study_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Clientes ordenados por probabilidad", fontsize=12)
    ax.set_ylabel("Ganancia acumulada", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Mostrar la leyenda con el punto óptimo incluido
    ax.legend(fontsize=9, ncol=2) 
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))

    # AJUSTES DE LÍMITES 
    ax.set_xlim(right=x_max)
    ax.set_ylim(bottom=0) 
    
    plt.tight_layout()

    # Guardar imagen en carpetas de Git
    os.makedirs("resultados/plots", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = f"resultados/plots/{study_name}_comparativo_semillas_{timestamp}.png"
    plt.savefig(ruta, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    logger.info(f"✅ Gráfico comparativo guardado: {ruta}")

    # Guardar imagen en Bckts
    os.makedirs(f"../../../buckets/b1/Compe_02/{study_name}", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = f"../../../buckets/b1/Compe_02/{study_name}/{study_name}_comparativo_semillas_{timestamp}.png"
    plt.savefig(ruta, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    logger.info(f"✅ Gráfico comparativo guardado: {ruta}")


    return {
        "ruta_grafico": ruta,
        "ganancias_max": ganancias_max,
        "curva_prom": curva_prom,
        "curva_std": curva_std,
    }

def comparar_semillas_en_grafico_con_ensamble(df_fe, mejores_params, semillas, mes_test, meses_train, study_name="multi_seed"):

    logger.info(f"=== Comparando {len(semillas)} semillas ===")

    curvas = []
    ganancias_max = []
    todas_predicciones = []
    y_test_global = None

    for seed in semillas:
        logger.info(f"🔁 Semilla: {seed}")
        np.random.seed(seed)
        random.seed(seed)

        resultados_test, y_pred_proba, y_test = evaluar_en_test(
            df_fe,
            mejores_params,
            mes_test=mes_test,
            meses_train=meses_train,
            seed=seed)

        y_test = np.asarray(y_test)
        y_pred_proba = np.asarray(y_pred_proba)

        if y_test_global is None:
            y_test_global = y_test

        todas_predicciones.append(y_pred_proba)

        ganancias_acumuladas, _, _ = calcular_ganancia_acumulada_optimizada(y_test, y_pred_proba)
        curvas.append(ganancias_acumuladas)
        ganancias_max.append(np.max(ganancias_acumuladas))

    # Calcular curva del ensamble total
    y_pred_ensamble = np.mean(todas_predicciones, axis=0)
    curva_ensamble, _, _ = calcular_ganancia_acumulada_optimizada(y_test_global, y_pred_ensamble)

    # Calcular resumen del ensamble
    y_pred_binary = (y_pred_ensamble > UMBRAL).astype(int)
    ganancia_test = ganancia_evaluator(y_test_global, y_pred_binary)
    predicciones_positivas = int(np.sum(y_pred_binary))
    porcentaje_positivas = float(np.mean(y_pred_binary) * 100)

    resultados_ensamble = {
        'ganancia_test': float(ganancia_test),
        'predicciones_positivas': predicciones_positivas,
        'porcentaje_positivas': porcentaje_positivas
    }

    # Emparejar largo de las curvas
    min_len = min(len(c) for c in curvas + [curva_ensamble])
    curvas = np.array([c[:min_len] for c in curvas])
    curva_ensamble = curva_ensamble[:min_len]

    curva_prom = curvas.mean(axis=0)
    curva_std = curvas.std(axis=0)

    LIMITE_X = 20000
    x_max = min(min_len, LIMITE_X)
    curva_prom_cortada = curva_prom[:x_max]
    idx_max = np.argmax(curva_prom_cortada)
    ganancia_optima = curva_prom_cortada[idx_max]
    clientes_optimos = idx_max

    # === GRAFICAR ===
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(12, 7))

    for i, curva in enumerate(curvas):
        ax.plot(curva[:x_max], alpha=0.8, lw=1.8, label=f"Seed {semillas[i]}")

    ax.plot(curva_prom[:x_max], color="black", lw=3, label="Promedio", zorder=5)
    ax.fill_between(range(x_max), 
                    curva_prom[:x_max] - curva_std[:x_max], 
                    curva_prom[:x_max] + curva_std[:x_max], 
                    color="gray", alpha=0.2)

    ax.plot(curva_ensamble[:x_max], color="blue", lw=3, linestyle="--", label="Ensamble Total", zorder=4)

    ax.scatter(clientes_optimos, ganancia_optima, 
               color='red', s=100, marker='*', zorder=10, 
               label=f"Óptimo Promedio: {ganancia_optima:,.0f} ({clientes_optimos} Clientes)")

    ax.set_title(f"Ganancia acumulada por semilla - {study_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Clientes ordenados por probabilidad", fontsize=12)
    ax.set_ylabel("Ganancia acumulada", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))
    ax.set_xlim(right=x_max)
    ax.set_ylim(bottom=0)
    plt.tight_layout()


    # Guardar gráfico
    os.makedirs("resultados/plots", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = f"resultados/plots/{study_name}_comparativo_semillas_{timestamp}.png"
    ruta_2 = f"../../../buckets/b1/Compe_02/{study_name}/{study_name}_{timestamp}.png"
    os.makedirs(os.path.dirname(ruta_2), exist_ok=True)



    plt.savefig(ruta_2, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(ruta, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"✅ Gráfico comparativo guardado: {ruta_2}")
    logger.info(f"✅ Gráfico comparativo guardado: {ruta}")

    return {
        "ruta_grafico": ruta,
        "ganancias_max": ganancias_max,
        "curva_prom": curva_prom,
        "curva_std": curva_std,
        "curva_ensamble": curva_ensamble,
        "resultados_ensamble": resultados_ensamble
    }




