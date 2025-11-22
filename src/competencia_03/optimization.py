import optuna
import lightgbm as lgb
import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
from config import *
from gain_function import calcular_ganancia, ganancia_lgb_binary, ganancia_evaluator, calcular_ganancia_top_k
from final_training import *

logging.basicConfig(
    level=logging.INFO,  # para ver mensajes DEBUG o superiores
    format='%(asctime)s - %(levelname)s - %(message)s',
)

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

        ganancia_total = ganancia_evaluator(y_val, y_pred_binary)
        ganancias.append(ganancia_total)

    # Promedio de ganancias
    ganancia_promedio = np.mean(ganancias)

    # Guardar en JSON y loggear
    guardar_iteracion(trial, ganancia_promedio)
    logger.debug(
        f"Trial {trial.number}: Ganancias = {[int(g) for g in ganancias]} | Promedio = {ganancia_promedio:,.0f}"
    )

    return ganancia_promedio



def objetivo_ganancia_ensamble(trial, df, undersampling=0.2) -> float:
    """
    Igual que objetivo_ganancia, pero entrena un modelo por cada semilla y ensambla
    promediando las predicciones en validación antes de calcular la ganancia.
    """

    # Hiperparámetros y configuración general
    semillas = SEMILLA
    mes_train = MES_TRAIN
    mes_validacion = MES_VALIDACION

    # Dividir datos en train y validación
    df_train = df[df['foto_mes'].isin(mes_train)].copy()
    df_val = df[df['foto_mes'].isin(mes_validacion)].copy()

    # --- UNDERSAMPLING A NIVEL CLIENTE ---
    if isinstance(undersampling, float) and 0 < undersampling < 1:
        np.random.seed(SEMILLA[0])
    
        # Log de distribución por mes antes de filtrar
        logger.info("Distribución de target por mes antes de undersampling:")
        for mes in sorted(df_train["foto_mes"].unique()):
            df_mes = df_train[df_train["foto_mes"] == mes]
            dist = df_mes["target"].value_counts().to_dict()
            logger.info(f"  Mes {mes}: {dist}")
    
        # Clientes con al menos un target=1
        clientes_con_target1 = df_train[df_train["target"] == 1]["numero_de_cliente"].unique()
        logger.info(f"Clientes con target=1: {len(clientes_con_target1)}")
    
        # Clientes que nunca tuvieron target=1
        todos_clientes = df_train["numero_de_cliente"].unique()
        clientes_siempre_0 = np.setdiff1d(todos_clientes, clientes_con_target1)
        logger.info(f"Clientes siempre 0: {len(clientes_siempre_0)}")
    
        # Subsamplear clientes 0
        n_subsample = int(len(clientes_siempre_0) * undersampling)
        clientes_siempre_0_sample = np.random.choice(clientes_siempre_0, n_subsample, replace=False)
        logger.info(f"Clientes 0 seleccionados por undersampling: {len(clientes_siempre_0_sample)}")
    
        # Combinar ambos grupos
        clientes_final = np.concatenate([clientes_con_target1, clientes_siempre_0_sample])
        logger.info(f"Total clientes en train después de undersampling: {len(clientes_final)}")
    
        # Filtrar df_train
        df_train = df_train[df_train["numero_de_cliente"].isin(clientes_final)].copy()
    
        # Log de distribución final
        logger.info("Distribución de target en df_train final:")
        dist_final = df_train["target"].value_counts().to_dict()
        logger.info(f"  0: {dist_final.get(0, 0)}, 1: {dist_final.get(1, 0)}")
        logger.info(f"  Proporción positivos: {dist_final.get(1, 0) / df_train.shape[0] * 100:.2f}%")
    
        # Log por mes final
        logger.info("Distribución por mes en df_train final:")
        for mes in sorted(df_train["foto_mes"].unique()):
            df_mes = df_train[df_train["foto_mes"] == mes]
            dist = df_mes["target"].value_counts().to_dict()
            logger.info(f"  Mes {mes}: {dist}")
    
    else:
        logger.info("Sin undersampling: se usan todos los clientes en train.")

    # Separar características y target
    X_train = df_train.drop(columns=['target', 'target_to_calculate_gan'])
    y_train = df_train['target']

    X_val = df_val.drop(columns=['target', 'target_to_calculate_gan'])
    y_val = df_val['target_to_calculate_gan']

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

    # --- ENTRENAMIENTO MULTISEMILLA CON ENSAMBLE ---
    preds_acumuladas = np.zeros(len(X_val))

    for seed in SEMILLA:
        params = params_base.copy()
        params['seed'] = seed

        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

        gbm = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_val],
            valid_names=['train', 'valid'],
            feval=ganancia_evaluator,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50),
            ],
        )

        preds_acumuladas += gbm.predict(X_val, num_iteration=gbm.best_iteration)

    # Promediar predicciones
    y_pred_proba_prom = preds_acumuladas / len(SEMILLA)


    # Calcular ganancia final del ensamble
    # ganancia_total = ganancia_evaluator(y_pred_proba_prom, y_val)
    
    ganancia_total = calcular_ganancia_top_k(y_val, y_pred_proba_prom, k=10000)
    
    # Normalizar por cantidad de meses usados
    n_meses_train = len(set(MES_TRAIN))
    # n_meses_valid = len(set(MES_VALIDACION))  # o len(MES_VALIDACION) si usás más de uno
    n_meses_valid = len(MES_VALIDACION) if isinstance(MES_VALIDACION, (list, tuple, set)) else 1

    # debug rápido (ejecutar dentro de objetivo_ganancia_ensamble después de obtener ganancia_total)
    print("DEBUG SANITY CHECK")
    print("MES_VALIDACION:", MES_VALIDACION)
    print("n_meses_valid:", len(MES_VALIDACION) if isinstance(MES_VALIDACION, (list,tuple,set)) else 1)
    print("Ganancia total (raw):", ganancia_total)
    print("Ganancia normalized (current):", ganancia_total / (len(MES_VALIDACION) if isinstance(MES_VALIDACION, (list,tuple,set)) else 1))


    ganancia_normalizada = ganancia_total / (n_meses_valid)
    


    # Guardar en JSON y loggear
    guardar_iteracion(trial, ganancia_normalizada)
    logger.debug(
        f"Trial {trial.number}: Ganancia (ensamble) = {ganancia_normalizada:,.0f}"
    )

    return ganancia_normalizada





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
    
    n_trials_previos = len([t for t in study.trials if t.value is not None])

    if n_trials_previos > 0:
        print("Trials previos válidos:", n_trials_previos)
        print("Best trial hasta ahora:", study.best_value)
    else:
        print("Aún no hay trials válidos en este estudio")


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
        study.optimize(lambda trial: objetivo_ganancia_por_meses(trial, df, undersampling), n_trials=trials_a_ejecutar)
        logger.info(f"🏆 Mejor ganancia: {study.best_value:,.0f}")
        logger.info(f"Mejores parámetros: {study.best_params}")
    else:
        logger.info(f"✅ Ya se completaron {n_trials} trials")
  
    return study




def crear_o_cargar_estudio(study_name: str = None, semilla: int = None) -> optuna.Study:
    """
    Crea un nuevo estudio de Optuna o carga uno existente de manera segura.

    Args:
        study_name: Nombre del estudio (si es None, usa STUDY_NAME del config)
        semilla: Semilla para reproducibilidad

    Returns:
        optuna.Study: Estudio de Optuna (nuevo o cargado)
    """
    # Usar valores por defecto del config si no se pasan
    study_name = study_name or STUDY_NAME
    semilla = semilla or (SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA)

    # Crear carpeta para bases de datos si no existe
    path_db = os.path.join(BUCKET_NAME, "optuna_db")
    os.makedirs(path_db, exist_ok=True)

    # Ruta completa de la base de datos
    db_file = os.path.join(path_db, f"{study_name}.db")
    storage = f"sqlite:///{db_file}"

    logger.info(f"📁 Usando storage: {storage}")

    # Crear o cargar el estudio de manera segura
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=semilla),
        storage=storage,
        load_if_exists=True  # <--- evita DuplicatedStudyError
    )

    valid_trials = [t for t in study.trials if t.value is not None]
    if valid_trials:
        logger.info(f"🏆 Mejor ganancia hasta ahora: {study.best_value:,.0f}")
    else:
        logger.info("⚠️ No hay trials válidos aún para mostrar mejor valor")


    return study



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




def objetivo_ganancia_por_meses(trial, df, undersampling=0.2) -> float:
    """
    Objetivo de optimización: entrenar por grupos+semillas y validar en abril y junio.
    Devuelve la ganancia promedio del ensamble global en ambos meses.
    """

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

    # --- Sugerir hiperparámetros ---
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

    ganancias_por_mes = []

    # --- Validar en abril y junio ---
    for mes, grupos_train, mes_predic in [
        ("abril", FINAL_TRAINING_GROUPS_APRIL, FINAL_PREDIC_APRIL),
        ("junio", FINAL_TRAINING_GROUPS_JUNE, FINAL_PREDIC_JUNE),
    ]:
        logger.info(f"=== OPTIMIZACIÓN VALIDANDO EN {mes.upper()} ===")

        grupos_datos = preparar_datos_entrenamiento_por_grupos_por_semilla(
            df,
            grupos_train,
            mes_predic,
            undersampling_ratio=undersampling,
            semillas=SEMILLA
        )

        modelos_por_grupo = entrenar_modelos_por_grupo_y_semilla(
            grupos_datos,
            mejores_params=params_base
        )

        df_predict = df[df["foto_mes"] == mes_predic]
        X_predict = df_predict.drop(columns=["target", "target_to_calculate_gan"])
        clientes_predict = df_predict["numero_de_cliente"].values

        resultados = generar_predicciones_finales(
            modelos_por_grupo,
            X_predict,
            clientes_predict,
            df_predict,
            top_k=TOP_K,
            mes=mes_predic
        )

        # ✅ Usar la ganancia del ensamble global
        ganancias_df = resultados["ganancias"]
        ganancia_mes = ganancias_df.loc[
            ganancias_df["grupo"] == "GLOBAL", "ganancia_test"
        ].iloc[0]

        ganancias_por_mes.append(ganancia_mes)
        logger.info(f"Ganancia {mes}: {ganancia_mes:,.0f}")

    # --- Promedio de ganancias ---
    ganancia_promedio = np.mean(ganancias_por_mes)

    guardar_iteracion(trial, ganancia_promedio)
    logger.debug(f"Trial {trial.number}: Ganancia promedio (abril+junio) = {ganancia_promedio:,.0f}")

    return ganancia_promedio




## Codigo Muerto


    # def lr_schedule(iteration):
    #     return params_base["lr_init"] * (params_base["lr_decay"] ** iteration)