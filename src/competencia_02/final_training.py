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

def entrenar_modelo_final(X_train: pd.DataFrame, y_train: pd.Series, mejores_params: dict) -> lgb.Booster:
    """
    Entrena el modelo final con los mejores hiperparámetros.
  
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        mejores_params: Mejores hiperparámetros de Optuna
  
    Returns:
        lgb.Booster: Modelo entrenado
    """
    logger.info("Iniciando entrenamiento del modelo final")
  
    # Configurar parámetros del modelo

    # def lr_schedule(iteration):
    #     return params_base["lr_init"] * (params_base["lr_decay"] ** iteration)
        
    params = {
        'objective': 'binary',
        'metric': 'None',  # Usamos nuestra métrica personalizada
        'random_state': SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA,
        'verbose': -1,
        'extra_trees': False,  # Para mayor diversidad entre semillas
        **mejores_params  # Agregar los mejores hiperparámetros
    }
  
    logger.info(f"Parámetros del modelo: {params}")
  
    # Crear dataset de LightGBM

    lgb_train = lgb.Dataset(X_train, label=y_train)
  
    # Entrenar modelo con lgb.train()
    modelo = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train],
        callbacks=[
        # lgb.reset_parameter(learning_rate=lr_schedule),
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=100)], 
        feval=ganancia_evaluator
    )
    return modelo

# def generar_predicciones_finales(modelo: lgb.Booster, X_predict: pd.DataFrame, clientes_predict: np.ndarray, umbral: float = 0.04) -> pd.DataFrame:
#     """
#     Genera las predicciones finales para el período objetivo.
  
#     Args:
#         modelo: Modelo entrenado
#         X_predict: Features para predicción
#         clientes_predict: IDs de clientes
#         umbral: Umbral para clasificación binaria
  
#     Returns:
#         pd.DataFrame: DataFrame con numero_cliente y predict
#     """
#     logger.info("Generando predicciones finales")
  
#     # Generar probabilidades con el modelo entrenado
#     y_pred = modelo.predict(X_predict, num_iteration=modelo.best_iteration)

#     # Convertir a predicciones binarias con el umbral establecido
#     y_pred_binary = (y_pred > umbral).astype(int)

#     # Crear DataFrame de 'resultados' con nombres de atributos que pide kaggle
#     resultados = pd.DataFrame({
#         'numero_de_cliente': clientes_predict,
#         'predict': y_pred_binary
#     })

#     # Estadísticas de predicciones
#     total_predicciones = len(resultados)
#     predicciones_positivas = (resultados['predict'] == 1).sum()
#     porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100

#     feature_importance(modelo)
  
#     logger.info(f"Predicciones generadas:")
#     logger.info(f"  Total clientes: {total_predicciones:,}")
#     logger.info(f"  Predicciones positivas: {predicciones_positivas:,} ({porcentaje_positivas:.2f}%)")
#     logger.info(f"  Predicciones negativas: {total_predicciones - predicciones_positivas:,}")
#     logger.info(f"  Umbral utilizado: {umbral}")
  
#     return resultados

def generar_predicciones_finales(
    modelo: lgb.Booster,
    X_predict: pd.DataFrame,
    clientes_predict: np.ndarray,
    umbral: float = 0.04,
    top_k: int = 10000
) -> dict:
    """
    Genera las predicciones finales para el período objetivo, tanto con umbral como con top_k.

    Args:
        modelo: Modelo entrenado
        X_predict: Features para predicción
        clientes_predict: IDs de clientes
        umbral: Umbral para clasificación binaria
        top_k: Cantidad de clientes con mayor probabilidad a seleccionar (opcional)

    Returns:
        dict: {'umbral': DataFrame, 'top_k': DataFrame (si aplica)}
    """
    logger.info("Generando predicciones finales")
    import os
    os.makedirs("predict", exist_ok=True)

    # Generar probabilidades con el modelo entrenado
    y_pred = modelo.predict(X_predict, num_iteration=modelo.best_iteration)

    # --- Predicciones con umbral ---
    y_pred_binary = (y_pred > umbral).astype(int)
    resultados_umbral = pd.DataFrame({
        'numero_de_cliente': clientes_predict,
        'predict': y_pred_binary,
        'probabilidad': y_pred
    })

    total_predicciones = len(resultados_umbral)
    predicciones_positivas = (resultados_umbral['predict'] == 1).sum()
    porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100

    feature_importance(modelo)

    logger.info(f"Predicciones con umbral generadas:")
    logger.info(f"  Total clientes: {total_predicciones:,}")
    logger.info(f"  Positivas: {predicciones_positivas:,} ({porcentaje_positivas:.2f}%)")
    logger.info(f"  Umbral utilizado: {umbral}")

    resultados = {'umbral': resultados_umbral[['numero_de_cliente', 'predict']]}

    # --- Predicciones con top_k ---
    if top_k is not None:
        logger.info(f"Generando predicciones con top_k={top_k:,}")
        df_topk = pd.DataFrame({
            'numero_de_cliente': clientes_predict,
            'probabilidad': y_pred
        }).sort_values('probabilidad', ascending=False)

        df_topk['predict'] = 0
        df_topk.iloc[:top_k, df_topk.columns.get_loc('predict')] = 1


        resultados['top_k'] = df_topk[['numero_de_cliente', 'predict']]

        logger.info(f"  Predicciones top_k generadas (K={top_k:,})")
        logger.info(f"  Máxima probabilidad: {df_topk['probabilidad'].iloc[0]:.4f}")
        logger.info(f"  Mínima probabilidad dentro del top_k: {df_topk['probabilidad'].iloc[top_k - 1]:.4f}")

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