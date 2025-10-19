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
    X_train = df_train.drop(columns=['target','target_to_calculate_gan])
    y_train = df_train['target']

    X_val = df_val.drop(columns=['target','target_to_calculate_gan'])
    y_val = df_val['target']

    # Crear datasets de LightGBM
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    
    # Valores por defecto (fallback)
    DEFAULT_HYPERPARAMS = {
    "num_leaves":      {"min": 5, "max": 50, "type": "int"},
    "learning_rate":   {"min": 0.005, "max": 0.10, "type": "float"},
    "min_data_in_leaf":{"min": 300, "max": 800, "type": "int"},
    "feature_fraction":{"min": 0.1, "max": 0.8, "type": "float"},
    "bagging_fraction":{"min": 0.2, "max": 0.8, "type": "float"},
    }

    # Merge entre lo que viene del YAML y los defaults
    PARAM_RANGES = {**DEFAULT_HYPERPARAMS, **HYPERPARAM_RANGES}

    params = {
        'objective': 'binary',
        'metric': 'custom',
        'boosting_type': 'gbdt',
        'first_metric_only': True,
        'boost_from_average': True,
        'feature_pre_filter': False,
        'max_bin': 31,
        'seed': semillas[0],
        'verbose': -1,
    }

    # usar los rangos de PARAM_RANGES
    for hp, cfg in PARAM_RANGES.items():
        if cfg["type"] == "int":
            params[hp] = trial.suggest_int(hp, cfg["min"], cfg["max"])
        elif cfg["type"] == "float":
            params[hp] = trial.suggest_float(hp, cfg["min"], cfg["max"])
        else:
            raise ValueError(f"Tipo de hiperparámetro no soportado: {cfg['type']}")



    # Entrenar modelo con función de ganancia personalizada
    gbm = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'valid'],
        feval=ganancia_evaluator,
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



def objetivo_ganancia_cv(trial, df) -> float:
    """
    Función objetivo para Optuna: maximiza ganancia en mes de validación.
    - Usa configuración YAML para períodos y semilla.
    - Define parámetros de LightGBM con rangos de YAML (con fallback).
    - Entrena modelo con métrica de ganancia personalizada.
    """

    # Configuración desde YAML
    semillas = SEMILLA
    mes_train = MES_TRAIN
    mes_validacion = MES_VALIDACION

    # Dividir datos
    df_train = df[df['foto_mes'].isin(mes_train + [mes_validacion])]


    X_train = df_train.drop(columns=['target', 'foto_mes','target_to_calculate_gan'])
    y_train = df_train['target']

    # Datasets de LightGBM
    lgb_train = lgb.Dataset(X_train, label=y_train)

    # Defaults por si no están en el YAML
    DEFAULT_HYPERPARAMS = {
        "num_leaves":      {"min": 5, "max": 50, "type": "int"},
        "learning_rate":   {"min": 0.005, "max": 0.10, "type": "float"},
        "min_data_in_leaf":{"min": 300, "max": 800, "type": "int"},
        "feature_fraction":{"min": 0.1, "max": 0.8, "type": "float"},
        "bagging_fraction":{"min": 0.2, "max": 0.8, "type": "float"},
        
    }

    # Merge YAML + defaults
    PARAM_RANGES = {**DEFAULT_HYPERPARAMS, **HYPERPARAM_RANGES}

    # Parámetros base de LightGBM
    params = {
        'objective': 'binary',
        'metric': 'custom',
        'boosting_type': 'gbdt',
        'first_metric_only': True,
        'boost_from_average': True,
        'feature_pre_filter': False,
        'max_bin': 31,
        'seed': semillas[0],
        'verbose': -1
    }

    # Agregar hiperparámetros desde YAML/defaults
    for hp, cfg in PARAM_RANGES.items():
        if cfg["type"] == "int":
            params[hp] = trial.suggest_int(hp, cfg["min"], cfg["max"])
        elif cfg["type"] == "float":
            params[hp] = trial.suggest_float(hp, cfg["min"], cfg["max"])
        else:
            raise ValueError(f"Tipo de hiperparámetro no soportado: {cfg['type']}")

    # Entrenamiento con función de ganancia personalizada
    cv_results = lgb.cv(
        params,
        lgb_train,
        feval=ganancia_evaluator,
        nfold=5,
        stratified=True,
        shuffle=True,
        num_boost_round=1000,
        seed= SEMILLA[0],
        callbacks=[
            lgb.early_stopping(stopping_rounds=25),
            lgb.log_evaluation(period=25)
        ],
    )
    
    print(cv_results.keys())


    # Extraer resultados de CV
    
    # To get the best mean gain:
    ganancias_cv = cv_results['valid ganancia-mean']
    if not isinstance(ganancias_cv, list):
        ganancias_cv = list(ganancias_cv)
    ganancia_maxima = max(ganancias_cv)
    best_iteration = len(ganancias_cv) - 1
    
    # Loggers debug
    logger.debug(f"Trial {trial.number}: Ganancia CV: {ganancia_maxima:,.0f}")
    logger.debug(f"Trial {trial.number}: Mejor Iteracion: {best_iteration}")

    guardar_iteracion_cv(trial,ganancia_maxima,ganancias_cv)

    return ganancia_maxima


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




### Optimizar

def optimizar_con_cv(df, n_trials=50) -> optuna.Study:
    """
    Ejecuta optimización bayesiana con Cross Validation usando Optuna.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe con los datos (incluye target y foto_mes).
    n_trials : int, default=50
        Número de pruebas/iteraciones que ejecutará Optuna.

    Returns
    -------
    optuna.Study
        Objeto Study de Optuna con los resultados de la optimización.
    """

    study_name = STUDY_NAME

    logger.info(f"Iniciando optimización con CV = {n_trials} trials")
    logger.info(
        f"Configuración CV: periodos = {MES_TRAIN + [MES_VALIDACION] if isinstance(MES_TRAIN, list) else [MES_TRAIN, MES_VALIDACION]}"
    )

    # Crear estudio de Optuna
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=SEMILLA[0]),
    )

    # Ejecutar optimización
    study.optimize(
        lambda trial: objetivo_ganancia_cv(trial, df),
        n_trials=n_trials,
    )

    # Resultados
    logger.info("Optimización con CV completada")
    logger.info(f"Mejor ganancia promedio: {study.best_value:,.0f}")
    logger.info(f"Mejores parámetros: {study.best_params}")
    logger.info(f"Total trials: {len(study.trials)}")

    return study



def crear_o_cargar_estudio(study_name: str = None, semilla: int = None) -> optuna.Study:
    """
    Crea un nuevo estudio de Optuna o carga uno existente desde SQLite.
  
    Args:
        study_name: Nombre del estudio (si es None, usa STUDY_NAME del config)
        semilla: Semilla para reproducibilidad
  
    Returns:
        optuna.Study: Estudio de Optuna (nuevo o cargado)
    """
    study_name = STUDY_NAME
  
    if semilla is None:
        semilla = SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA
  
    # Crear carpeta para bases de datos si no existe
    path_db = os.path.join(BUCKET_NAME, "optuna_db")
    os.makedirs(path_db, exist_ok=True)
  
    # Ruta completa de la base de datos
    db_file = os.path.join(path_db, f"{study_name}.db")
    storage = f"sqlite:///{db_file}"
  
    # Verificar si existe un estudio previo
    if os.path.exists(db_file):
        logger.info(f"⚡ Base de datos encontrada: {db_file}")
        logger.info(f"🔄 Cargando estudio existente: {study_name}")
  
        try:
            #PRESTAR ATENCION Y RAZONAR!!!
            study = optuna.load_study(study_name=study_name, storage=storage)
            n_trials_previos = len(study.trials)
  
            logger.info(f"✅ Estudio cargado exitosamente")
            logger.info(f"📊 Trials previos: {n_trials_previos}")
  
            if n_trials_previos > 0:
                logger.info(f"🏆 Mejor ganancia hasta ahora: {study.best_value:,.0f}")
  
            return study
  
        except Exception as e:
            logger.warning(f"⚠️ No se pudo cargar el estudio: {e}")
            logger.info(f"🆕 Creando nuevo estudio...")
    else:
        logger.info(f"🆕 No se encontró base de datos previa")
        logger.info(f"📁 Creando nueva base de datos: {db_file}")
  
    # Crear nuevo estudio
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=SEMILLA[0]),
    )

    # Ejecutar optimización
    study.optimize(
        lambda trial: objetivo_ganancia(trial, df, undersampling = 1.0),
        n_trials=n_trials,
    )

    logger.info(f"✅ Nuevo estudio creado: {study_name}")
    logger.info(f"💾 Storage: {storage}")
  
    return study


