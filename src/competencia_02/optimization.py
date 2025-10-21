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

def objetivo_ganancia(trial, df, undersampling=0.2) -> float:
    """
    Parameters:
    trial: trial de optuna
    df: dataframe con datos
    undersampling: float en (0,1) o False. Proporción de clientes con target=0 a mantener.

    Description:
    Función objetivo que maximiza ganancia en mes de validación.
    Utiliza configuración YAML para períodos y semilla.
    Define parámetros para el modelo LightGBM.
    Prepara dataset para entrenamiento y validación, aplicando undersampling a nivel cliente.
    Entrena 10 modelos (uno por cada semilla) con función de ganancia personalizada.
    Devuelve la ganancia promedio entre las semillas.
    """

    # Hiperparámetros y configuración general
    semillas = SEMILLA
    mes_train = MES_TRAIN
    mes_validacion = MES_VALIDACION

    # Dividir datos en train y validación
    df_train = df[df['foto_mes'].isin(mes_train)].copy()
    df_val = df[df['foto_mes'].isin([mes_validacion])].copy()

    # --- UNDERSAMPLING A NIVEL CLIENTE ---
    if isinstance(undersampling, float) and 0 < undersampling < 1:
        np.random.seed(SEMILLA[0])

        # Clientes que alguna vez tuvieron target=1 → conservar todos sus registros
        clientes_con_target1 = (
            df_train.groupby("numero_de_cliente")["target"]
            .max()
            .reset_index()
        )
        clientes_con_target1 = clientes_con_target1[
            clientes_con_target1["target"] == 1
        ]["numero_de_cliente"]

        # Clientes que siempre fueron 0
        clientes_siempre_0 = (
            df_train.loc[
                ~df_train["numero_de_cliente"].isin(clientes_con_target1),
                "numero_de_cliente",
            ]
            .unique()
        )

        # Subsamplear clientes 0
        n_subsample = int(len(clientes_siempre_0) * undersampling)
        clientes_siempre_0_sample = np.random.choice(
            clientes_siempre_0, n_subsample, replace=False
        )

        # Combinar ambos grupos
        clientes_final = np.concatenate(
            [clientes_con_target1.values, clientes_siempre_0_sample]
        )

        # Filtrar train
        df_train = df_train[df_train["numero_de_cliente"].isin(clientes_final)]

        logger.debug(
            f"Undersampling aplicado: {len(clientes_con_target1)} clientes con target=1 "
            f"+ {len(clientes_siempre_0_sample)} clientes 0 (de {len(clientes_siempre_0)} posibles) "
            f"→ total {len(clientes_final)} clientes en train."
        )

    else:
        logger.debug("Sin undersampling: se usan todos los clientes en train.")

    # Separar características y target
    X_train = df_train.drop(columns=['target', 'target_to_calculate_gan'])
    y_train = df_train['target']

    X_val = df_val.drop(columns=['target', 'target_to_calculate_gan'])
    y_val = df_val['target']

    # Rango por defecto de hiperparámetros
    DEFAULT_HYPERPARAMS = {
        "num_leaves":      {"min": 5, "max": 50, "type": "int"},
        "learning_rate":   {"min": 0.005, "max": 0.10, "type": "float"},
        "min_data_in_leaf":{"min": 300, "max": 800, "type": "int"},
        "feature_fraction":{"min": 0.1, "max": 0.8, "type": "float"},
        "bagging_fraction":{"min": 0.2, "max": 0.8, "type": "float"},
    }

    # Merge entre YAML y defaults
    PARAM_RANGES = {**DEFAULT_HYPERPARAMS, **HYPERPARAM_RANGES}

    # Sugerir hiperparámetros desde Optuna
    params_base = {
        'objective': 'binary',
        'metric': 'custom',
        'boosting_type': 'gbdt',
        'first_metric_only': True,
        'boost_from_average': True,
        'feature_pre_filter': False,
        'max_bin': 31,
        'verbose': -1,
    }

    for hp, cfg in PARAM_RANGES.items():
        if cfg["type"] == "int":
            params_base[hp] = trial.suggest_int(hp, cfg["min"], cfg["max"])
        elif cfg["type"] == "float":
            params_base[hp] = trial.suggest_float(hp, cfg["min"], cfg["max"])
        else:
            raise ValueError(f"Tipo de hiperparámetro no soportado: {cfg['type']}")

    # --- ENTRENAMIENTO MULTISEMILLA ---
    ganancias = []

    for seed in SEMILLA:
        params = params_base.copy()
        params['seed'] = seed

        # Crear datasets LightGBM
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

        # Entrenar modelo
        gbm = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_val],
            valid_names=['train', 'valid'],
            feval=ganancia_evaluator,
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50),
            ],
        )

        # Predicción y ganancia
        y_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)
        y_pred_binary = (y_pred > UMBRAL).astype(int)

        _, ganancia_total, _ = ganancia_evaluator(y_val, y_pred_binary)
        ganancias.append(ganancia_total)

    # Promedio de ganancias
    ganancia_promedio = np.mean(ganancias)

    # Guardar en JSON y loggear
    guardar_iteracion(trial, ganancia_promedio)
    logger.debug(
        f"Trial {trial.number}: Ganancias = {[int(g) for g in ganancias]} | Promedio = {ganancia_promedio:,.0f}"
    )

    return ganancia_promedio




def optimizar(df: pd.DataFrame, n_trials: int, study_name: str = None, undersampling: float = 0.2) -> optuna.Study:
    """
    Args:
        df: DataFrame con datos
        n_trials: Número de trials a ejecutar
        study_name: Nombre del estudio (si es None, usa el de config.yaml)
        undersampling: Undersampling para entrenamiento
  
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

    logger.info(f"Iniciando optimización con {n_trials} trials")
    logger.info(f"Configuración: TRAIN={MES_TRAIN}, VALID={MES_VALIDACION}, SEMILLA={SEMILLA}")
  
    # Crear o cargar estudio desde DuckDB
    study = crear_o_cargar_estudio(study_name, SEMILLA)

    # Calcular cuántos trials faltan
    trials_previos = len(study.trials)
    trials_a_ejecutar = max(0, n_trials - trials_previos)
  
    if trials_previos > 0:
        logger.info(f"🔄 Retomando desde trial {trials_previos}")
        logger.info(f"📝 Trials a ejecutar: {trials_a_ejecutar} (total objetivo: {n_trials})")
    else:
        logger.info(f"🆕 Nueva optimización: {n_trials} trials")
  
    # Ejecutar optimización
    if trials_a_ejecutar > 0:
        ##LO UNICO IMPORTANTE DEL METODO Y EL study CLARO
        study.optimize(lambda trial: objetivo_ganancia(trial, df, undersampling), n_trials=trials_a_ejecutar)
        logger.info(f"🏆 Mejor ganancia: {study.best_value:,.0f}")
        logger.info(f"Mejores parámetros: {study.best_params}")
    else:
        logger.info(f"✅ Ya se completaron {n_trials} trials")
  
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

