import pandas as pd
import lightgbm as lgb
import numpy as np
import logging
import os
from datetime import datetime
from config import FINAL_TRAIN, FINAL_PREDIC, SEMILLA, STUDY_NAME
from best_params import cargar_mejores_hiperparametros
from gain_function import ganancia_lgb_binary, ganancia_evaluator

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


def generar_predicciones_finales(
    modelos: list[lgb.Booster],
    X_predict: pd.DataFrame,
    clientes_predict: np.ndarray,
    umbral: float = 0.04,
    top_k: int = 10000
) -> dict:
    """
    Genera las predicciones finales promediando varios modelos (ensamble).
    Produce tanto predicciones con umbral como por top_k.

    Args
    ----
    modelos : list[lgb.Booster]
        Lista de modelos LightGBM entrenados.
    X_predict : pd.DataFrame
        Features para predicción.
    clientes_predict : np.ndarray
        IDs de clientes.
    umbral : float, default=0.04
        Umbral para clasificación binaria.
    top_k : int, default=10000
        Cantidad de clientes con mayor probabilidad a seleccionar.

    Returns
    -------
    dict
        {'umbral': DataFrame, 'top_k': DataFrame}
    """
    import os
    os.makedirs("predict", exist_ok=True)

    logger.info("=== GENERANDO PREDICCIONES FINALES (ENSAMBLE) ===")
    n_modelos = len(modelos)
    logger.info(f"Se detectaron {n_modelos} modelos para el ensamble.")

    # --- Promediar predicciones ---
    preds_sum = np.zeros(len(X_predict), dtype=np.float32)
    for i, modelo in enumerate(modelos, start=1):
        pred_i = modelo.predict(X_predict, num_iteration=modelo.best_iteration)
        preds_sum += pred_i
        logger.info(f"  Modelo {i}/{n_modelos} procesado.")
    y_pred = preds_sum / n_modelos

    # --- Predicciones binarias (umbral) ---
    y_pred_bin = (y_pred > umbral).astype(int)
    resultados_umbral = pd.DataFrame({
        "numero_de_cliente": clientes_predict,
        "predict": y_pred_bin,
        "probabilidad": y_pred
    })

    total = len(resultados_umbral)
    positivos = (resultados_umbral["predict"] == 1).sum()
    pct_positivos = positivos / total * 100
    logger.info(f"Total clientes: {total:,}")
    logger.info(f"Predicciones positivas: {positivos:,} ({pct_positivos:.2f}%)")
    logger.info(f"Umbral utilizado: {umbral}")

    # --- Feature importance del primer modelo (referencia) ---
    feature_importance(modelos[0])

    resultados = {"umbral": resultados_umbral[["numero_de_cliente", "predict"]]}

    # --- Predicciones por top_k ---
    logger.info(f"Generando predicciones con top_k={top_k:,}")
    df_topk = resultados_umbral[["numero_de_cliente", "probabilidad"]].copy()
    df_topk = df_topk.sort_values("probabilidad", ascending=False, ignore_index=True)
    df_topk["predict"] = 0
    df_topk.loc[:top_k - 1, "predict"] = 1

    resultados["top_k"] = df_topk[["numero_de_cliente", "predict"]]

    logger.info(f"Máx prob: {df_topk['probabilidad'].iloc[0]:.4f}")
    logger.info(f"Mín prob dentro del top_k: {df_topk['probabilidad'].iloc[top_k - 1]:.4f}")
    logger.info("✅ Predicciones finales generadas correctamente.")

    return resultados


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