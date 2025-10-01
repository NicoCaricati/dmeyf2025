import optuna
import lightgbm as lgb
import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
from config import *
from gain_function import calcular_ganancia, ganancia_lgb_binary

logger = logging.getLogger(__name__)

def objetivo_ganancia(trial, df) -> float:
        
    """
    Parameters:
    trial: trial de optuna
    df: dataframe con datos

    Description:
    Función objetivo que maximiza ganancia en mes de validación.
    Utiliza configuración YAML para períodos y semilla.
    Define parametros para el modelo LightGBM
    Preparar dataset para entrenamiento y validación
    Entrena modelo con función de ganancia personalizada
    Predecir y calcular ganancia
    Guardar cada iteración en JSON

    Returns:
    float: ganancia total
    """
    # Hiperparámetros a optimizar
    semillas = SEMILLA  # Desde configuración YAML
    mes_train = MES_TRAIN  # Desde configuración YAML
    mes_validacion = MES_VALIDACION  # Desde configuración YAML

    # Dividir datos en train y validación según meses
    df_train = df[df['foto_mes'].isin(mes_train)]
    df_val = df[df['foto_mes'].isin([mes_validacion])]

    # Sepa rar características y target
    X_train = df_train.drop(columns=['target', 'foto_mes', 'numero_de_cliente'])
    y_train = df_train['target']

    X_val = df_val.drop(columns=['target', 'foto_mes', 'numero_de_cliente'])
    y_val = df_val['target']

    # Crear datasets de LightGBM
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    num_leaves = trial.suggest_int('num_leaves', 5, 70)
    learning_rate = trial.suggest_float('learning_rate', 0.005, 0.15)  # más bajo, más iteraciones
    min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 300, 1000)
    feature_fraction = trial.suggest_float('feature_fraction', 0.2, 0.8)
    bagging_fraction = trial.suggest_float('bagging_fraction', 0.5, 1.0)
    

    params = {
    'objective': 'binary',
    'metric': 'custom',
    'boosting_type': 'gbdt',
    'first_metric_only': True,
    'boost_from_average': True,
    'feature_pre_filter': False,
    'max_bin': 31,
    'num_leaves': num_leaves,
    'learning_rate': learning_rate,
    'min_data_in_leaf': min_data_in_leaf,
    'feature_fraction': feature_fraction,
    'bagging_fraction': bagging_fraction,
    'seed': semillas[0],
    'verbose': -1
    }   



    # Entrenar modelo con función de ganancia personalizada
    gbm = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'valid'],
        feval=ganancia_lgb_binary,
        num_boost_round=1000,
        callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)],  # opcional, logs cada 50 rounds
    )

    # Predecir en conjunto de validación
    y_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)

    # Convertir probabilidades a predicciones binarias
    y_pred_binary = (y_pred > UMBRAL).astype(int)


    ganancia_total = calcular_ganancia(y_val, y_pred_binary)

    # Guardar cada iteración en JSON
    guardar_iteracion(trial, ganancia_total)


    logger.debug(f"Trial {trial.number}: Ganancia = {ganancia_total:,.0f}")

    return ganancia_total


### Guardar Iteracion

def guardar_iteracion(trial, ganancia, archivo_base=None):
    """
    Guarda cada iteración de la optimización en un único archivo JSON.
  
    Args:
        trial: Trial de Optuna
        ganancia: Valor de ganancia obtenido
        archivo_base: Nombre base del archivo (si es None, usa el de config.yaml)
    """
    if archivo_base is None:
        archivo_base = STUDY_NAME
  
    # Nombre del archivo único para todas las iteraciones
    archivo = f"resultados/{archivo_base}_iteraciones.json"
  
    # Datos de esta iteración
    iteracion_data = {
        'trial_number': trial.number,
        'params': trial.params,
        'value': float(ganancia),
        'datetime': datetime.now().isoformat(),
        'state': 'COMPLETE',  # Si llegamos aquí, el trial se completó exitosamente
        'configuracion': {
            'semilla': SEMILLA,
            'mes_train': MES_TRAIN,
            'mes_validacion': MES_VALIDACION
        }
    }
  
    # Cargar datos existentes si el archivo ya existe
    if os.path.exists(archivo):
        with open(archivo, 'r') as f:
            try:
                datos_existentes = json.load(f)
                if not isinstance(datos_existentes, list):
                    datos_existentes = []
            except json.JSONDecodeError:
                datos_existentes = []
    else:
        datos_existentes = []
  
    # Agregar nueva iteración
    datos_existentes.append(iteracion_data)
  
    # Guardar todas las iteraciones en el archivo
    with open(archivo, 'w') as f:
        json.dump(datos_existentes, f, indent=2)
  
    logger.info(f"Iteración {trial.number} guardada en {archivo}")
    logger.info(f"Ganancia: {ganancia:,.0f}" + "---" + "Parámetros: {params}")


### Optimizar

def optimizar(df, n_trials=30) -> optuna.Study:
    """
    Args:
        df: DataFrame con datos
        n_trials: Número de trials a ejecutar
        study_name: Nombre del estudio (si es None, usa el de config.yaml)
  
    Description:
       Ejecuta optimización bayesiana de hiperparámetros usando configuración YAML.
       Guarda cada iteración en un archivo JSON separado. 
       Pasos:
        1. Crear estudio de Optuna
        2. Ejecutar optimización
        3. Retornar estudio

    Returns:
        optuna.Study: Estudio de Optuna con resultados
    """

    study_name = STUDY_NAME
        
    # Crear estudio de Optuna
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name
    )

    # Ejecutar optimización
    study.optimize(
        lambda trial: objetivo_ganancia(trial, df),
        n_trials=n_trials
    )


    logger.info(f"Iniciando optimización con {n_trials} trials")
    logger.info(f"Configuración: TRAIN={MES_TRAIN}, VALID={MES_VALIDACION}, SEMILLA={SEMILLA}")

  
    # Resultados
    logger.info(f"Mejor ganancia: {study.best_value:,.0f}")
    logger.info(f"Mejores parámetros: {study.best_params}")
  
  
    return study


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
    X_test = df_test.drop(columns=['target', 'foto_mes', 'numero_de_cliente'])
    X_train = df_train_completo.drop(columns=['target', 'foto_mes', 'numero_de_cliente'])
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
        num_boost_round=1000,
        feval=ganancia_lgb_binary,
        callbacks=[
            lgb.log_evaluation(period=50)
        ],
    )

    # Predecir en conjunto de test
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    y_pred_binary = (y_pred > UMBRAL).astype(int)

    # Calcular solo la ganancia
    ganancia_test = calcular_ganancia(y_test, y_pred_binary)
  
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
  
    return resultados

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



