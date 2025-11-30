# final training

import pandas as pd
import lightgbm as lgb
import numpy as np
import logging
import os
from datetime import datetime
from config import *
from best_params import cargar_mejores_hiperparametros
from gain_function import ganancia_lgb_binary, ganancia_evaluator, calcular_ganancia_top_k
from typing import Tuple
from undersampling import undersample_clientes


logger = logging.getLogger(__name__)

def preparar_datos_entrenamiento_final(df: pd.DataFrame) -> tuple:
    """
    Prepara los datos para el entrenamiento final usando todos los períodos de FINAL_TRAIN.
  
    Args:
        df: DataFrame con todos los datos
  
    Returns:
        tuple: (X_train, y_train, X_predict, clientes_predict)
    """
    logger.info(f"Preparando datos para entrenamiento final")
    logger.info(f"Períodos de entrenamiento: {FINAL_TRAIN}")
    logger.info(f"Período de predicción: {FINAL_PREDIC}")
  
    # Datos de entrenamiento: todos los períodos en FINAL_TRAIN
  
    # Datos de predicción: período FINAL_PREDIC 

    # logger.info(f"Registros de entrenamiento: {len(df_train):,}")
    # logger.info(f"Registros de predicción: {len(df_predict):,}")
  
    #Corroborar que no esten vacios los df

    # Preparar features y target para entrenamiento
    df_entrenamiento_final = df
    df_train = df_entrenamiento_final[df_entrenamiento_final['foto_mes'].isin(FINAL_TRAIN)]
    df_predict = df_entrenamiento_final[df_entrenamiento_final['foto_mes'] == FINAL_PREDIC]
    #filtro los meses de train para entrenar el modelo final, y predigo en test
    y_train = df_train['target']
    X_train = df_train.drop(columns=['target','target_to_calculate_gan'])
    y_predict = df_predict['target']
    X_predict = df_predict.drop(columns=['target','target_to_calculate_gan'])

    # Preparar features para predicción
    clientes_predict = df_predict['numero_de_cliente'].values
    features_cols = X_train.columns.tolist()

    logger.info(f"Features utilizadas: {len(features_cols)}")
    logger.info(f"Distribución del target - 0: {(y_train == 0).sum():,}, 1: {(y_train == 1).sum():,}")
  
    return X_train, y_train, X_predict, clientes_predict

def entrenar_modelo_final(X_train: pd.DataFrame,
                                   y_train: pd.Series,
                                   X_predict: pd.DataFrame,
                                   mejores_params: dict,
                                   semillas: list[int]) -> tuple:
    """
    Entrena múltiples modelos LightGBM (uno por semilla) y promedia sus predicciones.
    Devuelve las probabilidades promedio y los modelos entrenados.

    Args
    ----
    X_train, y_train : datos de entrenamiento
    X_predict : features del set de predicción final
    mejores_params : dict
        Hiperparámetros óptimos de Optuna
    semillas : list[int]
        Lista de semillas a utilizar para el ensamble

    Returns
    -------
    tuple[np.ndarray, list[lgb.Booster]]
        (predicciones_promedio, lista_de_modelos)
    """
    logger.info("=== ENTRENAMIENTO FINAL (ENSEMBLE DE SEMILLAS) ===")
    logger.info(f"Semillas utilizadas: {semillas}")
    logger.info(f"Tamaño del set de entrenamiento: {len(X_train):,}")
    logger.info(f"Tamaño del set de predicción: {len(X_predict):,}")

    modelos = []
    preds_acumuladas = np.zeros(len(X_predict))

    for seed in semillas:
        logger.info(f"Entrenando modelo con semilla {seed}...")

        params = {
            'objective': 'binary',
            'metric': 'None',  # métrica custom
            'boosting_type': 'gbdt',
            'first_metric_only': True,
            'boost_from_average': True,
            'feature_pre_filter': False,
            'max_bin': 31,
            'seed': seed,
            'verbose': -1,
            **mejores_params
        }

        lgb_train = lgb.Dataset(X_train, label=y_train)

        modelo = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train],
            feval=ganancia_evaluator,
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=100)
            ]
        )

        modelos.append(modelo)
        preds = modelo.predict(X_predict, num_iteration=modelo.best_iteration)
        preds_acumuladas += preds

    # Promedio de probabilidades del ensamble
    preds_prom = preds_acumuladas / len(semillas)
    logger.info(f"✅ Ensamble final completado con {len(semillas)} modelos.")
    
    
    return preds_prom, modelos


# def generar_predicciones_finales(
#     modelos: list[lgb.Booster],
#     X_predict: pd.DataFrame,
#     clientes_predict: np.ndarray,
#     umbral: float = 0.04,
#     top_k: int = 10000
# ) -> dict:
#     """
#     Genera las predicciones finales promediando varios modelos (ensamble).
#     Produce tanto predicciones con umbral como por top_k.

#     Args
#     ----
#     modelos : list[lgb.Booster]
#         Lista de modelos LightGBM entrenados.
#     X_predict : pd.DataFrame
#         Features para predicción.
#     clientes_predict : np.ndarray
#         IDs de clientes.
#     umbral : float, default=0.04
#         Umbral para clasificación binaria.
#     top_k : int, default=10000
#         Cantidad de clientes con mayor probabilidad a seleccionar.

#     Returns
#     -------
#     dict
#         {'umbral': DataFrame, 'top_k': DataFrame}
#     """
#     import os
#     os.makedirs("predict", exist_ok=True)

#     logger.info("=== GENERANDO PREDICCIONES FINALES (ENSAMBLE) ===")
#     n_modelos = len(modelos)
#     logger.info(f"Se detectaron {n_modelos} modelos para el ensamble.")

#     # --- Promediar predicciones ---
#     preds_sum = np.zeros(len(X_predict), dtype=np.float32)
#     for i, modelo in enumerate(modelos, start=1):
#         pred_i = modelo.predict(X_predict, num_iteration=modelo.best_iteration)
#         preds_sum += pred_i
#         logger.info(f"  Modelo {i}/{n_modelos} procesado.")
#     y_pred = preds_sum / n_modelos

#     # --- Predicciones binarias (umbral) ---
#     y_pred_bin = (y_pred > umbral).astype(int)
#     resultados_umbral = pd.DataFrame({
#         "numero_de_cliente": clientes_predict,
#         "predict": y_pred_bin,
#         "probabilidad": y_pred
#     })

#     total = len(resultados_umbral)
#     positivos = (resultados_umbral["predict"] == 1).sum()
#     pct_positivos = positivos / total * 100
#     logger.info(f"Total clientes: {total:,}")
#     logger.info(f"Predicciones positivas: {positivos:,} ({pct_positivos:.2f}%)")
#     logger.info(f"Umbral utilizado: {umbral}")

#     # --- Feature importance del primer modelo (referencia) ---
#     feature_importance(modelos[0])

#     resultados = {"umbral": resultados_umbral[["numero_de_cliente", "predict"]]}

#     # --- Predicciones por top_k ---
#     logger.info(f"Generando predicciones con top_k={top_k:,}")
#     df_topk = resultados_umbral[["numero_de_cliente", "probabilidad"]].copy()
#     df_topk = df_topk.sort_values("probabilidad", ascending=False, ignore_index=True)
#     df_topk["predict"] = 0
#     df_topk.loc[:top_k - 1, "predict"] = 1

#     resultados["top_k"] = df_topk[["numero_de_cliente", "predict"]]

#     logger.info(f"Máx prob: {df_topk['probabilidad'].iloc[0]:.4f}")
#     logger.info(f"Mín prob dentro del top_k: {df_topk['probabilidad'].iloc[top_k - 1]:.4f}")
#     logger.info("✅ Predicciones finales generadas correctamente.")

#     return resultados


# def generar_predicciones_finales(
#     modelos_por_grupo: dict[str, list[lgb.Booster]],
#     X_predict: pd.DataFrame,
#     clientes_predict: np.ndarray,
#     df_predict: pd.DataFrame,
#     top_k: int = 10000
# ) -> dict:
#     """
#     Genera predicciones finales por top_k, guarda predicciones individuales con ganancia,
#     y produce dos ensambles: global y por grupo.
#     """
#     import os
#     os.makedirs("predict", exist_ok=True)

#     logger.info("=== GENERANDO PREDICCIONES FINALES (ENSAMBLE) ===")
#     predicciones_individuales = []
#     preds_sum_global = np.zeros(len(X_predict), dtype=np.float32)
#     n_modelos = 0

#     preds_por_grupo = []

#     for nombre_grupo, modelos in modelos_por_grupo.items():
#         logger.info(f"Procesando grupo '{nombre_grupo}' con {len(modelos)} modelos")
#         preds_grupo = np.zeros(len(X_predict), dtype=np.float32)

#         for i, modelo in enumerate(modelos, start=1):
#             pred_i = modelo.predict(X_predict, num_iteration=modelo.best_iteration)
#             preds_sum_global += pred_i
#             preds_grupo += pred_i
#             n_modelos += 1

#             df_i = pd.DataFrame({
#                 "numero_de_cliente": clientes_predict,
#                 "probabilidad": pred_i,
#                 "grupo": nombre_grupo,
#                 "modelo_id": f"{nombre_grupo}_seed{i}"
#             })
#             df_i["predict"] = 0
#             df_i = df_i.sort_values("probabilidad", ascending=False, ignore_index=True)
#             df_i.loc[:top_k - 1, "predict"] = 1

#             df_i = df_i.merge(df_predict[["numero_de_cliente", "target_to_calculate_gan"]], on="numero_de_cliente", how="left")
#             df_i["ganancia"] = df_i["predict"] * df_i["target_to_calculate_gan"]

#             ganancia_total = df_i["ganancia"].sum()
#             logger.info(f"Modelo {df_i['modelo_id'].iloc[0]}: Ganancia total = {ganancia_total:,.2f}")

#             predicciones_individuales.append(df_i[["numero_de_cliente", "probabilidad", "predict", "ganancia", "modelo_id", "grupo"]])

#         preds_grupo /= len(modelos)
#         preds_por_grupo.append(preds_grupo)

#     # Guardar CSV de predicciones individuales
#     df_all_preds = pd.concat(predicciones_individuales, ignore_index=True)
#     df_all_preds.to_csv("predict/predicciones_individuales.csv", index=False)

#     # Ensamble global
#     y_pred_global = preds_sum_global / n_modelos
#     df_topk_global = pd.DataFrame({
#         "numero_de_cliente": clientes_predict,
#         "probabilidad": y_pred_global
#     }).sort_values("probabilidad", ascending=False, ignore_index=True)
#     df_topk_global["predict"] = 0
#     df_topk_global.loc[:top_k - 1, "predict"] = 1

#     # Ensamble por grupo
#     y_pred_grupos = sum(preds_por_grupo) / len(preds_por_grupo)
#     df_topk_grupos = pd.DataFrame({
#         "numero_de_cliente": clientes_predict,
#         "probabilidad": y_pred_grupos
#     }).sort_values("probabilidad", ascending=False, ignore_index=True)
#     df_topk_grupos["predict"] = 0
#     df_topk_grupos.loc[:top_k - 1, "predict"] = 1

#     logger.info("✅ Predicciones finales generadas correctamente.")
#     return {
#         "top_k_global": df_topk_global[["numero_de_cliente", "predict"]],
#         "top_k_grupos": df_topk_grupos[["numero_de_cliente", "predict"]]
#     }



def feature_importance(modelo: lgb.Booster, max_num_features: int = 1000):
    """
    Muestra la importancia de las variables del modelo LightGBM.
  
    Args:
        modelo: Modelo entrenado
        max_num_features: Número máximo de features a mostrar
    """
    import matplotlib.pyplot as plt
    import os
    os.makedirs("feature_importance", exist_ok=True)
    fecha = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Obtener importancia de features
    importance_gain = modelo.feature_importance(importance_type='gain')
    importance_split = modelo.feature_importance(importance_type='split')
    feature_names = modelo.feature_name()
  
    # Crear DataFrame para visualización
    feat_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance_gain': importance_gain,
        'importance_split': importance_split
    }).sort_values(by='importance_gain', ascending=False)
    
    feat_imp_df.to_csv(f"feature_importance/feature_importance_{STUDY_NAME}_{fecha}.csv", index=False)
    logger.info(f"Importancia de las primeras {max_num_features} variables guardada en 'feature_importance/feature_importance_{STUDY_NAME}.csv'")



def entrenar_modelo_final_undersampling(X_train: pd.DataFrame,
                                        y_train: pd.Series,
                                        X_predict: pd.DataFrame,
                                        mejores_params: dict,
                                        semillas: list[int],
                                        ratio_undersampling: float = 0.2) -> Tuple[np.ndarray, list[lgb.Booster]]:
    """
    Entrena múltiples modelos LightGBM con undersampling por semilla y promedia sus predicciones.

    Args:
        X_train: Features del set de entrenamiento.
        y_train: Target binario.
        X_predict: Features del set de predicción final.
        mejores_params: Hiperparámetros óptimos.
        semillas: Lista de semillas para el ensamble.
        ratio_undersampling: Proporción de clientes 0 a conservar (entre 0 y 1).

    Returns:
        Tuple con (predicciones promedio, lista de modelos entrenados).
    """
    logger.info("=== ENTRENAMIENTO FINAL CON UNDERSAMPLING POR SEMILLA ===")
    logger.info(f"Semillas utilizadas: {semillas}")
    logger.info(f"Ratio de undersampling: {ratio_undersampling}")
    logger.info(f"Tamaño del set de predicción: {len(X_predict):,}")

    # Combinar X_train + y_train para aplicar undersampling
    df_train = X_train.copy()
    df_train["target"] = y_train
    if "numero_de_cliente" not in df_train.columns:
        raise ValueError("La columna 'numero_de_cliente' es requerida para el undersampling.")

    modelos = []
    preds_acumuladas = np.zeros(len(X_predict))

    for seed in semillas:
        logger.info(f"🔁 Entrenando modelo con semilla {seed}...")

        df_us = undersample_clientes(df_train, ratio=ratio_undersampling, semilla=seed)
        X_us = df_us.drop(columns=["target"])
        y_us = df_us["target"]

        logger.info(f"📊 Tamaño del set de entrenamiento (undersampled): {len(X_us):,}")
        logger.info(f"🎯 Distribución target: {y_us.value_counts().to_dict()}")

        params = {
            'objective': 'binary',
            'metric': 'None',
            'boosting_type': 'gbdt',
            'first_metric_only': True,
            'boost_from_average': True,
            'feature_pre_filter': False,
            'max_bin': 31,
            'seed': seed,
            'verbose': -1,
            **mejores_params
        }

        lgb_train = lgb.Dataset(X_us, label=y_us)

        modelo = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train],
            feval=ganancia_evaluator,
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=100)
            ]
        )

        modelos.append(modelo)
        preds = modelo.predict(X_predict, num_iteration=modelo.best_iteration)
        preds_acumuladas += preds

    preds_prom = preds_acumuladas / len(semillas)
    logger.info(f"✅ Ensamble final completado con {len(semillas)} modelos.")

    return preds_prom, modelos


def preparar_datos_entrenamiento_por_grupos(df: pd.DataFrame, grupos: dict[str, list[int]], final_predic: int) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """
    Prepara los datos de entrenamiento para cada grupo definido en FINAL_TRAINING_GROUPS.

    Returns
    -------
    dict[str, tuple[X_train, y_train]]
    """
    grupos_datos = {}
    for nombre, meses in grupos.items():
        df_train = df[df["foto_mes"].isin(meses)]
        X_train = df_train.drop(columns=["target", "target_to_calculate_gan"])
        y_train = df_train["target"]
        grupos_datos[nombre] = (X_train, y_train)

    logger.info(f"Datos preparados para {len(grupos_datos)} grupos.")
    return grupos_datos


# def entrenar_modelos_por_grupo(grupos_datos: dict[str, tuple[pd.DataFrame, pd.Series]],
#                                X_predict: pd.DataFrame,
#                                mejores_params: dict,
#                                semillas: list[int]) -> list[lgb.Booster]:
#     """
#     Entrena un modelo por grupo y por semilla. Devuelve todos los modelos entrenados.
#     """
#     modelos = []

#     for nombre_grupo, (X_train, y_train) in grupos_datos.items():
#         logger.info(f"=== Entrenando grupo '{nombre_grupo}' con {len(X_train):,} registros ===")
#         for seed in semillas:
#             logger.info(f"  Semilla {seed}")
#             params = {
#                 'objective': 'binary',
#                 'metric': 'None',
#                 'boosting_type': 'gbdt',
#                 'first_metric_only': True,
#                 'boost_from_average': True,
#                 'feature_pre_filter': False,
#                 'max_bin': 31,
#                 'seed': seed,
#                 'verbose': -1,
#                 **mejores_params
#             }

#             lgb_train = lgb.Dataset(X_train, label=y_train)

#             modelo = lgb.train(
#                 params,
#                 lgb_train,
#                 valid_sets=[lgb_train],
#                 feval=ganancia_evaluator,
#                 callbacks=[
#                     lgb.early_stopping(stopping_rounds=100),
#                     lgb.log_evaluation(period=100)
#                 ]
#             )

#             modelos.append(modelo)

#     logger.info(f"✅ Entrenamiento completado: {len(modelos)} modelos generados.")
#     return modelos


def entrenar_modelos_por_grupo(grupos_datos: dict[str, tuple[pd.DataFrame, pd.Series]],
                               mejores_params: dict,
                               semillas: list[int]) -> dict[str, list[lgb.Booster]]:
    """
    Entrena modelos por grupo y semilla. Devuelve dict: grupo → lista de modelos.
    """
    modelos_por_grupo = {}

    for nombre_grupo, (X_train, y_train) in grupos_datos.items():
        logger.info(f"=== Entrenando grupo '{nombre_grupo}' con {len(X_train):,} registros ===")
        modelos = []
        for seed in semillas:
            logger.info(f"  Semilla {seed}")
            params = {
                'objective': 'binary',
                'metric': 'None',
                'boosting_type': 'gbdt',
                'first_metric_only': True,
                'boost_from_average': True,
                'feature_pre_filter': False,
                'max_bin': 31,
                'seed': seed,
                'verbose': -1,
                **mejores_params
            }

            lgb_train = lgb.Dataset(X_train, label=y_train)

            modelo = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_train],
                feval=ganancia_evaluator,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100),
                    lgb.log_evaluation(period=100)
                ]
            )
            modelos.append(modelo)

        modelos_por_grupo[nombre_grupo] = modelos

    logger.info(f"✅ Entrenamiento completado: {sum(len(m) for m in modelos_por_grupo.values())} modelos generados.")
    return modelos_por_grupo



def preparar_datos_entrenamiento_por_grupos_por_semilla(
    df: pd.DataFrame,
    grupos: dict[str, list[int]],
    final_predic: int,
    undersampling_ratio: float = 0.2,
    semillas: list[int] = [555557]
) -> dict[str, dict[int, tuple[pd.DataFrame, pd.Series]]]:
    grupos_datos = {}

    for nombre_grupo, meses in grupos.items():
        df_grupo = df[df["foto_mes"].isin(meses)]
        grupos_datos[nombre_grupo] = {}

        for seed in semillas:
            df_sampleado = undersample_clientes(df_grupo, ratio=undersampling_ratio, semilla=seed)
            X_train = df_sampleado.drop(columns=["target", "target_to_calculate_gan"])
            y_train = df_sampleado["target"]
            grupos_datos[nombre_grupo][seed] = (X_train, y_train)

            logger.info(f"Grupo '{nombre_grupo}' con semilla {seed}: {len(X_train):,} registros")

    logger.info(f"✅ Datos preparados para {len(grupos_datos)} grupos y {len(semillas)} semillas por grupo.")
    return grupos_datos



def entrenar_modelos_por_grupo_y_semilla(
    grupos_datos: dict[str, dict[int, tuple[pd.DataFrame, pd.Series]]],
    mejores_params: dict
) -> dict[str, list[lgb.Booster]]:
    modelos_por_grupo = {}

    for nombre_grupo, semillas_dict in grupos_datos.items():
        modelos_por_grupo[nombre_grupo] = []

        for seed, (X_train, y_train) in semillas_dict.items():
            logger.info(f"Entrenando grupo '{nombre_grupo}' con semilla {seed}")
            params = {
                'objective': 'binary',
                'metric': 'None',
                'boosting_type': 'gbdt',
                'first_metric_only': True,
                'boost_from_average': True,
                'feature_pre_filter': False,
                'max_bin': 31,
                'seed': seed,
                'verbose': -1,
                **mejores_params
            }

            lgb_train = lgb.Dataset(X_train, label=y_train)

            modelo = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_train],
                feval=ganancia_evaluator,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100),
                    lgb.log_evaluation(period=100)
                ]
            )

            modelos_por_grupo[nombre_grupo].append(modelo)

    logger.info(f"✅ Entrenamiento completado: {sum(len(m) for m in modelos_por_grupo.values())} modelos generados.")
    return modelos_por_grupo

import logging
import os
import numpy as np
import pandas as pd
import lightgbm as lgb

logger = logging.getLogger(__name__)

# def generar_predicciones_finales(
#     modelos_por_grupo: dict[str, list[lgb.Booster]],
#     X_predict: pd.DataFrame,
#     clientes_predict: np.ndarray,
#     df_predict: pd.DataFrame,
#     top_k: int = 10000
# ) -> dict:
#     os.makedirs("predict", exist_ok=True)

#     logger.info("Iniciando generación de predicciones finales...")
#     logger.info(f"Cantidad de clientes a predecir: {len(clientes_predict)}")
#     logger.info(f"Cantidad de grupos de modelos: {len(modelos_por_grupo)}")

#     predicciones_individuales = []
#     preds_sum_global = np.zeros(len(X_predict), dtype=np.float32)
#     n_modelos = 0
#     preds_por_grupo = []

#     for nombre_grupo, modelos in modelos_por_grupo.items():
#         logger.info(f"Procesando grupo: {nombre_grupo} con {len(modelos)} modelos")
#         preds_grupo = np.zeros(len(X_predict), dtype=np.float32)

#         for i, modelo in enumerate(modelos, start=1):
#             logger.info(f"Generando predicciones con modelo {i} del grupo {nombre_grupo}")
#             pred_i = modelo.predict(X_predict, num_iteration=modelo.best_iteration)
#             preds_sum_global += pred_i
#             preds_grupo += pred_i
#             n_modelos += 1

#             df_i = pd.DataFrame({
#                 "numero_de_cliente": clientes_predict,
#                 "probabilidad": pred_i,
#                 "grupo": nombre_grupo,
#                 "modelo_id": f"{nombre_grupo}_seed{i}"
#             })
#             df_i["predict"] = 0
#             df_i = df_i.sort_values("probabilidad", ascending=False, ignore_index=True)
#             df_i.loc[:top_k - 1, "predict"] = 1

#             if "target_to_calculate_gan" in df_predict.columns:
#                 df_i = df_i.merge(
#                     df_predict[["numero_de_cliente", "target_to_calculate_gan"]],
#                     on="numero_de_cliente",
#                     how="left"
#                 )
#                 df_i["ganancia"] = df_i["predict"] * df_i["target_to_calculate_gan"]

#             predicciones_individuales.append(
#                 df_i[["numero_de_cliente", "probabilidad", "predict", "grupo", "modelo_id"] + 
#                      (["ganancia"] if "ganancia" in df_i.columns else [])]
#             )

#         preds_grupo /= len(modelos)
#         preds_por_grupo.append(preds_grupo)

#     if predicciones_individuales:
#         df_all_preds = pd.concat(predicciones_individuales, ignore_index=True)
#         df_all_preds.to_csv("predict/predicciones_individuales.csv", index=False)
#         logger.info(f"CSV de predicciones individuales guardado con {len(df_all_preds)} filas")
#     else:
#         logger.warning("No se generaron predicciones individuales (¿modelos_por_grupo vacío?)")

#     if n_modelos == 0:
#         logger.error("No se entrenó ningún modelo, no se pueden generar predicciones globales")
#         return {}

#     # Global
#     y_pred_global = preds_sum_global / n_modelos
#     df_topk_global = pd.DataFrame({
#         "numero_de_cliente": clientes_predict,
#         "probabilidad": y_pred_global
#     }).sort_values("probabilidad", ascending=False, ignore_index=True)
#     df_topk_global["predict"] = 0
#     df_topk_global.loc[:top_k - 1, "predict"] = 1
#     df_topk_global.to_csv("predict/predicciones_global.csv", index=False)
#     logger.info(f"CSV global guardado con {len(df_topk_global)} filas")

#     # Grupos
#     y_pred_grupos = sum(preds_por_grupo) / len(preds_por_grupo)
#     df_topk_grupos = pd.DataFrame({
#         "numero_de_cliente": clientes_predict,
#         "probabilidad": y_pred_grupos
#     }).sort_values("probabilidad", ascending=False, ignore_index=True)
#     df_topk_grupos["predict"] = 0
#     df_topk_grupos.loc[:top_k - 1, "predict"] = 1
#     df_topk_grupos.to_csv("predict/predicciones_grupos.csv", index=False)
#     logger.info(f"CSV de grupos guardado con {len(df_topk_grupos)} filas")

#     logger.info("Generación de predicciones finales completada exitosamente")
    
#     return {
#         "top_k_global": df_topk_global,
#         "top_k_grupos": df_topk_grupos
#     }


def generar_predicciones_finales(
    modelos_por_grupo: dict[str, list[lgb.Booster]],
    X_predict: pd.DataFrame,
    clientes_predict: np.ndarray,
    df_predict: pd.DataFrame,
    top_k: int = 10000,
    mes: int | None = None
) -> dict:
    os.makedirs("predict", exist_ok=True)

    logger.info("Iniciando generación de predicciones finales...")
    logger.info(f"Cantidad de clientes a predecir: {len(clientes_predict)}")
    logger.info(f"Cantidad de grupos de modelos: {len(modelos_por_grupo)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    predicciones_individuales = []
    resultados_ganancias = []   # <-- aquí guardamos las métricas
    preds_sum_global = np.zeros(len(X_predict), dtype=np.float32)
    n_modelos = 0
    preds_por_grupo = []

    # Etiquetas verdaderas (para calcular ganancia)
    y_true = df_predict["target_to_calculate_gan"].values

    for nombre_grupo, modelos in modelos_por_grupo.items():
        logger.info(f"Procesando grupo: {nombre_grupo} con {len(modelos)} modelos")
        preds_grupo = np.zeros(len(X_predict), dtype=np.float32)

        for i, modelo in enumerate(modelos, start=1):
            logger.info(f"Generando predicciones con modelo {i} del grupo {nombre_grupo}")
            y_pred_proba = modelo.predict(X_predict, num_iteration=modelo.best_iteration)
            preds_sum_global += y_pred_proba
            preds_grupo += y_pred_proba
            n_modelos += 1

            # Guardar predicciones individuales
            df_i = pd.DataFrame({
                "numero_de_cliente": clientes_predict,
                "probabilidad": y_pred_proba,
                "grupo": nombre_grupo,
                "modelo_id": f"{nombre_grupo}_seed{i}"
            })
            df_i["predict"] = 0
            df_i = df_i.sort_values("probabilidad", ascending=False, ignore_index=True)
            df_i.loc[:top_k - 1, "predict"] = 1

            if "target_to_calculate_gan" in df_predict.columns:
                df_i = df_i.merge(
                    df_predict[["numero_de_cliente", "target_to_calculate_gan"]],
                    on="numero_de_cliente",
                    how="left"
                )
                df_i["ganancia"] = df_i["predict"] * df_i["target_to_calculate_gan"]

            predicciones_individuales.append(df_i)

            # === Calcular ganancia con calcular_ganancia_top_k ===
            ganancia_test = calcular_ganancia_top_k(y_true,y_pred_proba)
            resultados_ganancias.append({
                "mes": mes,
                "grupo": nombre_grupo,
                "modelo_id": f"{nombre_grupo}_seed{i}",
                "ganancia_test": float(ganancia_test)
            })

        preds_grupo /= len(modelos)
        preds_por_grupo.append(preds_grupo)

    # # Guardar predicciones individuales
    # if predicciones_individuales:
    #     df_all_preds = pd.concat(predicciones_individuales, ignore_index=True)
    #     df_all_preds.to_csv("predict/predicciones_individuales.csv", index=False)
    #     logger.info(f"CSV de predicciones individuales guardado con {len(df_all_preds)} filas")

    # Global
    y_pred_global = preds_sum_global / n_modelos
    df_topk_global = pd.DataFrame({
        "numero_de_cliente": clientes_predict,
        "probabilidad": y_pred_global
    }).sort_values("probabilidad", ascending=False, ignore_index=True)
    df_topk_global["predict"] = 0
    df_topk_global.loc[:top_k - 1, "predict"] = 1
    df_topk_global.to_csv(f"predict/{STUDY_NAME}_predicciones_global_{mes}_{timestamp}.csv", index=False)

    # Ganancia global
    ganancia_global = calcular_ganancia_top_k(y_true,y_pred_global)
    resultados_ganancias.append({
        "mes": mes,
        "grupo": "GLOBAL",
        "modelo_id": "ensamble_global",
        "ganancia_test": float(ganancia_global)
    })

    # Grupos
    y_pred_grupos = sum(preds_por_grupo) / len(preds_por_grupo)
    df_topk_grupos = pd.DataFrame({
        "numero_de_cliente": clientes_predict,
        "probabilidad": y_pred_grupos
    }).sort_values("probabilidad", ascending=False, ignore_index=True)
    df_topk_grupos["predict"] = 0
    df_topk_grupos.loc[:top_k - 1, "predict"] = 1
    df_topk_grupos.to_csv(f"predict/{STUDY_NAME}_predicciones_grupos_{mes}_{timestamp}.csv", index=False)

    # Ganancia grupos
    ganancia_grupos = calcular_ganancia_top_k(y_true,y_pred_grupos)
    resultados_ganancias.append({
        "mes": mes,
        "grupo": "GRUPOS",
        "modelo_id": "ensamble_grupos",
        "ganancia_test": float(ganancia_grupos)
    })

    # Guardar CSV de ganancias
    df_ganancias = pd.DataFrame(resultados_ganancias)
    df_ganancias.to_csv(f"predict/ganancias_modelos_{mes}_{timestamp}.csv", index=False)
    df_ganancias.to_csv(f"../../../buckets/b1/Compe_02/{STUDY_NAME}/ganancias_modelos_{mes}_{timestamp}.csv", index=False)    
    logger.info(f"✅ CSV de ganancias guardado: predict/{STUDY_NAME}_ganancias_modelos_{mes}_{timestamp}.csv")

    return {
        "top_k_global": df_topk_global,
        "top_k_grupos": df_topk_grupos,
        "ganancias": df_ganancias
    }
