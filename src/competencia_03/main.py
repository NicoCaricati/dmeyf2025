# main


import logging
from datetime import datetime
import os
import pandas as pd
import numpy as np
import polars as pl
from config import *
import re
from snapshot import *

### Configuración de logging ###
fecha = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
nombre_log = f"/{STUDY_NAME}/log_{STUDY_NAME}_{fecha}.log"
bucket_name = BUCKET_NAME
os.makedirs(f"{bucket_name}/{STUDY_NAME}", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler(bucket_name + nombre_log),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Iniciando programa de optimización con log fechado")


from features import feature_engineering_lag, feature_engineering_delta, feature_engineering_regr_slope_window, feature_engineering_ratio, feature_engineering_tc_total, generar_ctrx_features, feature_engineering_cpayroll_trx_corregida, feature_engineering_mpayroll_corregida, variables_aux,feature_engineering_robust_by_month_polars,ajustar_por_ipc, detectar_grupo_excluido, detectar_variable_excluida, imputar_ceros_por_mes_anterior, generar_cambios_de_pendiente_multiples_fast, feature_engineering_delta_max, feature_engineering_delta_mean, imputar_ceros_por_promedio, transformar_a_percentil_rank, transformar_a_grupos_percentiles
from loader import cargar_datos, convertir_clase_ternaria_a_target
from optimization import *
from best_params import cargar_mejores_hiperparametros, obtener_estadisticas_optuna
from final_training import *
from output_manager import guardar_predicciones_finales
from test import *
from grafico_test import *
from evaluar_meses_test import evaluar_meses_test
from undersampling import undersample_clientes
from analisis_optuna import *


### Manejo de Configuración en YAML ###
logger.info("Configuración cargada desde YAML")
logger.info(f"STUDY_NAME: {STUDY_NAME}")
logger.info(f"DATA_PATH: {DATA_PATH}")
logger.info(f"SEMILLA: {SEMILLA}")
logger.info(f"MES_TRAIN: {MES_TRAIN}")
logger.info(f"MESES_OPTIMIZACION: {MESES_OPTIMIZACION}")
logger.info(f"MES_VALIDACION: {MES_VALIDACION}")
logger.info(f"MES_TEST: {MES_TEST}")
logger.info(f"FINAL_TRAIN: {FINAL_TRAIN}")
logger.info(f"FINAL_PREDIC: {FINAL_PREDIC}")
logger.info(f"GANANCIA_ACIERTO: {GANANCIA_ACIERTO}")
logger.info(f"COSTO_ESTIMULO: {COSTO_ESTIMULO}")
logger.info(f"UMBRAL: {UMBRAL}")
logger.info(f"HIPERPARAMETROS: {HYPERPARAM_RANGES}")
logger.info(f"UNDERSAMPLING_OPTIMIZACION: {UNDERSAMPLING_OPTIMIZACION}")
logger.info(f"UNDERSAMPLING_ENTRENAMIENTO_ENSAMBLE: {UNDERSAMPLING_ENTRENAMIENTO_ENSAMBLE}")




### Main ###
def main():
    """Pipeline principal con optimización usando configuración YAML."""
    logger.info("=== INICIANDO OPTIMIZACIÓN CON CONFIGURACIÓN YAML ===")

    crear_snapshot_modelo(STUDY_NAME)

    # path_parquet = os.path.join(BUCKET_NAME, "data", f"df_fe{STUDY_NAME}.parquet")
    path_parquet = "../../../buckets/b1/Compe_02/data/df_feRepito Modelo Final US 0.2 8 Semillas - Mejor Modelo + W3.parquet"

    if os.path.exists(path_parquet):
        logger.info("✅ df_fe.parquet encontrado")
        df_fe = pd.read_parquet(path_parquet)
    else:
        logger.info("❌ df_fe.parquet no encontrado")
        # 1. Cargar datos
        df = cargar_datos(DATA_PATH)
        if df is None:
            logger.error("No se pudieron cargar los datos; 'cargar_datos' retornó None.")
            raise ValueError("cargar_datos devolvió None. Verificar ruta o contenido de 'data/competencia_01_crudo.csv'.")
        logger.info(f"Datos cargados: {df.shape}")
        
        # variable_excluida = detectar_variable_excluida(STUDY_NAME)
        
        # if variable_excluida and variable_excluida in df.columns:
        #     df = df.drop(columns=[variable_excluida])
        #     logger.info(f"📉 Variable individual '{variable_excluida}' excluida del dataset.")

        # 1. Undersampling
        df_fe = convertir_clase_ternaria_a_target(df)
        # df_fe = df_fe[df_fe["target"].notnull()].copy()
        # df_fe = undersample_clientes(df_fe, UNDERSAMPLING, 555557)
        # logger.info(f"Después de undersampling: {df_fe.shape}")


        # Supongamos que ya tenés tu DataFrame "dataset"
        df_fe["ctrx_quarter_normalizado"] = df_fe["ctrx_quarter"].astype(float)
        
        df_fe.loc[df_fe["cliente_antiguedad"] == 1, "ctrx_quarter_normalizado"] *= 5.0
        df_fe.loc[df_fe["cliente_antiguedad"] == 2, "ctrx_quarter_normalizado"] *= 2.0
        df_fe.loc[df_fe["cliente_antiguedad"] == 3, "ctrx_quarter_normalizado"] *= 1.2
    
        # # 2. Feature Engineering
        # # Excluyo meses problematicos
        # meses_excluir = [201904, 201905, 201910, 202006]
        # df_fe = df_fe[~df_fe["foto_mes"].isin(meses_excluir)].copy()
        # logger.info(f"Después de excluir meses problemáticos: {df_fe.shape}")

        # Imputacion para corregir 0s
        df_fe = imputar_ceros_por_promedio(df_fe, columnas_no_imputar=['target','target_to_calculate_gan'])

        # Excluyo Comisiones Otras 
        df_fe = df_fe.drop(columns=['ccomisiones_otras','internet'])
        
        # # Agrego Variables para controlar mejor continuidad
        # df_fe = generar_ctrx_features(df_fe)        

        df_fe = feature_engineering_tc_total(df_fe)
        df_fe = variables_aux(df_fe)
        # Excluyo las variables no corregidas          
        cols_ajustar_ipc = [
            c for c in df_fe.columns
            if c.startswith(('m', 'Visa_m', 'Master_m','TC_Total_m')) and 'dolares' not in c
        ]
        df_fe = transformar_a_percentil_rank(df_fe, cols_ajustar_ipc, columna_mes='foto_mes')

        columnas_a_excluir = ["foto_mes","cliente_edad","numero_de_cliente","target","target_to_calculate_gan"]
        columnas_para_fe_regresiones = [
            c for c in df_fe.columns
            if c.startswith(('m', 'Visa_m', 'Master_m','TC_Total_m','Visa_F', 'Visa_f','Master_F', 'Master_f')) 
            and c not in columnas_a_excluir
        ]
        
        columnas_para_fe_deltas = [
            c for c in df_fe.columns
            if c.startswith(('c', 'Visa_c', 'Master_c','Master_s','Visa_s','TC_Total_c','TC_Total_s','t','Visa_F', 'Visa_f','Master_F', 'Master_f')) 
            and c not in columnas_a_excluir
        ]
        df_fe = df_fe.astype({col: "float32" for col in df_fe.select_dtypes("float").columns})
        # for i in (1,2):
        #     df_fe = feature_engineering_lag(df_fe, columnas=atributos, cant_lag=i)

        df_fe = generar_cambios_de_pendiente_multiples_fast(df_fe, columnas=columnas_para_fe_regresiones, ventana_corta=3, ventana_larga=6)
        df_fe = generar_cambios_de_pendiente_multiples_fast(df_fe, columnas=columnas_para_fe_regresiones, ventana_corta=1, ventana_larga=3)
        df_fe = df_fe.astype({col: "float32" for col in df_fe.select_dtypes("float").columns})  

        # df_fe = generar_cambios_de_pendiente_multiples_fast(df_fe, columnas=columnas_para_fe_regresiones, ventana_corta=6, ventana_larga=12)

        # for i in (2,3,6,8,10,12,15):
        #     df_fe = feature_engineering_regr_slope_window(df_fe, columnas=columnas_para_fe_regresiones, ventana = i)
        #     df_fe = df_fe.astype({col: "float32" for col in df_fe.select_dtypes("float").columns})
        for i in (2,3):
            df_fe = feature_engineering_delta(df_fe, columnas=columnas_para_fe_deltas, cant_delta = i)
        df_fe = df_fe.astype({col: "float32" for col in df_fe.select_dtypes("float").columns})  
        for i in (4,8):
            # df_fe = feature_engineering_delta_max(df_fe, columnas=columnas_para_fe_deltas, ventana=i)
            df_fe = feature_engineering_delta_mean(df_fe, columnas=columnas_para_fe_deltas, ventana=i)
        
        df_fe = df_fe.astype({col: "float32" for col in df_fe.select_dtypes("float").columns})  

        
        logger.info(f"Feature Engineering completado: {df_fe.shape}")
        
    
        df_fe.to_parquet(
            os.path.join(BUCKET_NAME, "data", f"df_fe{STUDY_NAME}.parquet"),
            compression='snappy'
        )    
    logger.info("⏳ CSV cargado o creado, ahora ejecutando optimización...")


    # 4. Ejecutar optimización (función simple)
    
    study = optimizar(df_fe, n_trials=50,study_name = STUDY_NAME ,undersampling = UNDERSAMPLING_OPTIMIZACION)
  
    # 5. Análisis adicional
    logger.info("=== ANÁLISIS DE RESULTADOS ===")

    analizar_resultados_optuna()
    
    trials_df = study.trials_dataframe()
    
    if trials_df is not None and len(trials_df) > 0:
        # Ordenar por valor (mayor ganancia)
        top_5 = trials_df.nlargest(5, 'value')
        logger.info("Top 5 mejores trials:")
    
        for idx, trial in top_5.iterrows():
            # Extraer parámetros (columnas que empiezan con 'params_')
            params_cols = [c for c in trial.index if c.startswith('params_')]
            if params_cols:
                params = {col.replace('params_', ''): trial[col] for col in params_cols}
            else:
                params = {}
    
            logger.info(
                f"Trial {int(trial['number'])}: "
                f"Ganancia = {trial['value']:,.0f} | "
                f"Parámetros: {params}"
            )
    else:
        logger.warning("No se encontraron trials para analizar.")

    # logger.info("=== OPTIMIZACIÓN COMPLETADA ===")

    #  05 Test en mes desconocido

    # Cargar mejores hiperparámetros

    mejores_params = cargar_mejores_hiperparametros()

    # mejores_params = {'bagging_fraction': 0.648239786, 'feature_fraction': 0.338110921, 'lambda_l1': 3.152084178, 'lambda_l2': 2.623895465, 'learning_rate': 0.074681467, 'min_data_in_leaf': 10, 'num_boost_round': 496, 'num_leaves': 26} # Opti sin US
    # mejores_params = {'num_leaves': 86, 'learning_rate': 0.04515219676722008, 'min_data_in_leaf': 45, 'feature_fraction': 0.2783670269042045, 'bagging_fraction': 0.68927175577007, 'lambda_l1': 1.4668038650423412, 'lambda_l2': 4.8010252173774495, 'num_boost_round': 507} # Opti con 0.2 de US
    # mejores_params = {'num_leaves': 71, 'learning_rate': 0.005943961863023024, 'min_data_in_leaf': 88, 'feature_fraction': 0.6094884732441374, 'bagging_fraction': 0.30532645375787404, 'lambda_l1': 0.1442564185202138, 'lambda_l2': 1.9492290528756926, 'num_boost_round': 497} # Opti con 0.5 de US
    # mejores_params =  {'num_leaves': 106, 'learning_rate': 0.05318395463346495, 'min_data_in_leaf': 4, 'feature_fraction': 0.49759556652323156, 'bagging_fraction': 0.7176155814161423, 'lambda_l1': 4.792320092280481, 'lambda_l2': 2.275425835398769, 'num_boost_round': 679} # Opti con 0.05 de US
    
    # Nueva Opti de 0.2
    # mejores_params = {'num_leaves': 121, 'learning_rate': 0.08944748172892189, 'min_data_in_leaf': 47, 'feature_fraction': 0.5831901957235187, 'bagging_fraction': 0.9395824062687965, 'lambda_l1': 4.4131882397060185, 'lambda_l2': 2.385519727758512, 'num_boost_round': 818}
    # mejores_params = {'num_leaves': 121, 'learning_rate': 0.08944748172892189, 'min_data_in_leaf': 47, 'feature_fraction': 0.5831901957235187, 'bagging_fraction': 0.9395824062687965, 'num_boost_round': 818}


    # mejores_params = {'bagging_fraction': 0.6714999873184199, 'feature_fraction': 0.31776519651631674, 'lambda_l1': 0.2944267005943108, 'lambda_l2': 2.4490808003374225, 'learning_rate': 0.03518640034343286, 'min_data_in_leaf': 47, 'neg_bagging_fraction': 0.16338893024243595, 'num_boost_round': 778, 'num_leaves': 84, 'pos_bagging_fraction': 0.40622347584093277} # Opti Nueva




    # logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")

    # df_fe_under = undersample_clientes(df_fe, UNDERSAMPLING, 555557)
    # df_fe_under = df_fe_under.select_dtypes(include=["number", "bool"]).copy()
    
    # # Evaluación multimes
    # evaluar_meses_test(
    #     df_fe=df_fe,
    #     mejores_params=mejores_params,
    #     semillas=SEMILLA,
    #     study_name=STUDY_NAME,
    #     config_meses=MESES_EVALUACION
    # )

  
    # === 06 Entrenar modelo final (distintos periodos) ===
    
    # Entrenamiento en Abril
    logger.info("=== ENTRENAMIENTO FINAL ABRIL ===")
    
    # Preparar datos por grupo y semilla con undersampling
    grupos_datos_abril = preparar_datos_entrenamiento_por_grupos_por_semilla(
        df_fe,
        FINAL_TRAINING_GROUPS_APRIL,
        FINAL_PREDIC_APRIL,
        undersampling_ratio=UNDERSAMPLING_ENTRENAMIENTO_ENSAMBLE,
        semillas=SEMILLA
    )
    
    # Preparar datos de predicción
    df_predict_abril = df_fe[df_fe["foto_mes"] == FINAL_PREDIC_APRIL]
    X_predict_abril = df_predict_abril.drop(columns=["target", "target_to_calculate_gan"])
    clientes_predict_abril = df_predict_abril["numero_de_cliente"].values
    
    # Entrenar modelos por grupo y semilla
    modelos_por_grupo_abril = entrenar_modelos_por_grupo_y_semilla(grupos_datos_abril, mejores_params)

    
    # Guardar importancia de features SOLO para la primera semilla de cada grupo
    for grupo, modelos in modelos_por_grupo_abril.items():
        if modelos:  # asegurarse de que haya modelos
            primer_modelo = modelos[0]  # tomar el de la primera semilla
            logger.info(f"📊 Guardando importancia de features para grupo {grupo}, primera semilla")
            feature_importance(primer_modelo)
        
        # Generar predicciones finales (ahora con mes)
        resultados_abril = generar_predicciones_finales(
            modelos_por_grupo_abril,
            X_predict_abril,
            clientes_predict_abril,
            df_predict_abril,
            top_k=TOP_K,
            mes=FINAL_PREDIC_APRIL
        )
        
    # Guardar predicciones
    guardar_predicciones_finales({"top_k": resultados_abril["top_k_global"]}, f"{FINAL_PREDIC_APRIL}_global")
    guardar_predicciones_finales({"top_k": resultados_abril["top_k_grupos"]}, f"{FINAL_PREDIC_APRIL}_grupos")
    
    # Guardar ganancias
    resultados_abril["ganancias"].to_csv(f"predict/ganancias_{STUDY_NAME}_{FINAL_PREDIC_APRIL}.csv", index=False)
    logger.info(f"✅ CSV de ganancias guardado: predict/ganancias_{STUDY_NAME}_{FINAL_PREDIC_APRIL}.csv")

    # Entrenamiento en Mayo
    logger.info("=== ENTRENAMIENTO FINAL MAYO ===")
    
    # Preparar datos por grupo y semilla con undersampling
    grupos_datos_mayo = preparar_datos_entrenamiento_por_grupos_por_semilla(
        df_fe,
        FINAL_TRAINING_GROUPS_MAYO,
        FINAL_PREDIC_MAYO,
        undersampling_ratio=UNDERSAMPLING_ENTRENAMIENTO_ENSAMBLE,
        semillas=SEMILLA
    )
    
    # Preparar datos de predicción
    df_predict_mayo = df_fe[df_fe["foto_mes"] == FINAL_PREDIC_MAYO]
    X_predict_mayo = df_predict_mayo.drop(columns=["target", "target_to_calculate_gan"])
    clientes_predict_mayo = df_predict_mayo["numero_de_cliente"].values
    
    # Entrenar modelos por grupo y semilla
    modelos_por_grupo_mayo = entrenar_modelos_por_grupo_y_semilla(grupos_datos_mayo, mejores_params)
    
    # Generar predicciones finales (ahora con mes)
    resultados_mayo = generar_predicciones_finales(
        modelos_por_grupo_mayo,
        X_predict_mayo,
        clientes_predict_mayo,
        df_predict_mayo,
        top_k=TOP_K,
        mes=FINAL_PREDIC_MAYO
    )
    
    # Guardar predicciones
    guardar_predicciones_finales({"top_k": resultados_mayo["top_k_global"]}, f"{FINAL_PREDIC_MAYO}_global")
    guardar_predicciones_finales({"top_k": resultados_mayo["top_k_grupos"]}, f"{FINAL_PREDIC_MAYO}_grupos")
    
    # Guardar ganancias
    resultados_mayo["ganancias"].to_csv(f"predict/ganancias_{STUDY_NAME}_{FINAL_PREDIC_MAYO}.csv", index=False)
    logger.info(f"✅ CSV de ganancias guardado: predict/ganancias_{STUDY_NAME}_{FINAL_PREDIC_MAYO}.csv")

    
    
        
    
    # Entrenamiento en Junio
    logger.info("=== ENTRENAMIENTO FINAL JUNIO ===")
    
    # Preparar datos por grupo y semilla con undersampling
    grupos_datos_junio = preparar_datos_entrenamiento_por_grupos_por_semilla(
        df_fe,
        FINAL_TRAINING_GROUPS_JUNE,
        FINAL_PREDIC_JUNE,
        undersampling_ratio=UNDERSAMPLING_ENTRENAMIENTO_ENSAMBLE,
        semillas=SEMILLA
    )
    
    # Preparar datos de predicción
    df_predict_junio = df_fe[df_fe["foto_mes"] == FINAL_PREDIC_JUNE]
    X_predict_junio = df_predict_junio.drop(columns=["target", "target_to_calculate_gan"])
    clientes_predict_junio = df_predict_junio["numero_de_cliente"].values
    
    # Entrenar modelos por grupo y semilla
    modelos_por_grupo_junio = entrenar_modelos_por_grupo_y_semilla(grupos_datos_junio, mejores_params)
    
    # Generar predicciones finales (ahora con mes)
    resultados_junio = generar_predicciones_finales(
        modelos_por_grupo_junio,
        X_predict_junio,
        clientes_predict_junio,
        df_predict_junio,
        top_k=TOP_K,
        mes=FINAL_PREDIC_JUNE
    )
    
    # Guardar predicciones
    guardar_predicciones_finales({"top_k": resultados_junio["top_k_global"]}, f"{FINAL_PREDIC_JUNE}_global")
    guardar_predicciones_finales({"top_k": resultados_junio["top_k_grupos"]}, f"{FINAL_PREDIC_JUNE}_grupos")
    
    # Guardar ganancias
    resultados_junio["ganancias"].to_csv(f"predict/ganancias_{STUDY_NAME}_{FINAL_PREDIC_JUNE}.csv", index=False)
    logger.info(f"✅ CSV de ganancias guardado: predict/ganancias_{STUDY_NAME}_{FINAL_PREDIC_JUNE}.csv")

    
    # Entrenamiento en Julio
    logger.info("=== ENTRENAMIENTO FINAL JULIO ===")
    
    # Preparar datos por grupo y semilla con undersampling
    grupos_datos_julio = preparar_datos_entrenamiento_por_grupos_por_semilla(
        df_fe,
        FINAL_TRAINING_GROUPS_JULIO,
        FINAL_PREDIC_JULIO,
        undersampling_ratio=UNDERSAMPLING_ENTRENAMIENTO_ENSAMBLE,
        semillas=SEMILLA
    )
    
    # Preparar datos de predicción
    df_predict_julio = df_fe[df_fe["foto_mes"] == FINAL_PREDIC_JULIO]
    X_predict_julio = df_predict_julio.drop(columns=["target", "target_to_calculate_gan"])
    clientes_predict_julio = df_predict_julio["numero_de_cliente"].values
    
    # Entrenar modelos por grupo y semilla
    modelos_por_grupo_julio = entrenar_modelos_por_grupo_y_semilla(grupos_datos_julio, mejores_params)
    
    # Generar predicciones finales (ahora con mes)
    resultados_julio = generar_predicciones_finales(
        modelos_por_grupo_julio,
        X_predict_julio,
        clientes_predict_julio,
        df_predict_julio,
        top_k=TOP_K,
        mes=FINAL_PREDIC_JULIO
    )
    
    # Guardar predicciones
    guardar_predicciones_finales({"top_k": resultados_julio["top_k_global"]}, f"{FINAL_PREDIC_JULIO}_global")
    guardar_predicciones_finales({"top_k": resultados_julio["top_k_grupos"]}, f"{FINAL_PREDIC_JULIO}_grupos")
    
    # Guardar ganancias
    resultados_julio["ganancias"].to_csv(f"predict/ganancias_{STUDY_NAME}_{FINAL_PREDIC_JULIO}.csv", index=False)
    logger.info(f"✅ CSV de ganancias guardado: predict/ganancias_{STUDY_NAME}_{FINAL_PREDIC_JULIO}.csv")



    
    # # Entrenamiento en Agosto
    # logger.info("=== ENTRENAMIENTO FINAL AGOSTO ===")
    
    # # Preparar datos por grupo y semilla con undersampling
    # grupos_datos_agosto = preparar_datos_entrenamiento_por_grupos_por_semilla(
    #     df_fe,
    #     FINAL_TRAINING_GROUPS_AGOSTO,
    #     FINAL_PREDIC_AGOSTO,
    #     undersampling_ratio=UNDERSAMPLING_ENTRENAMIENTO_ENSAMBLE,
    #     semillas=SEMILLA
    # )
    
    # # Preparar datos de predicción
    # df_predict_agosto = df_fe[df_fe["foto_mes"] == FINAL_PREDIC_AGOSTO]
    # X_predict_agosto = df_predict_agosto.drop(columns=["target", "target_to_calculate_gan"])
    # clientes_predict_agosto = df_predict_agosto["numero_de_cliente"].values
    
    # # Entrenar modelos por grupo y semilla
    # modelos_por_grupo_agosto = entrenar_modelos_por_grupo_y_semilla(grupos_datos_agosto, mejores_params)
    
    # # Generar predicciones finales (ahora con mes)
    # resultados_agosto = generar_predicciones_finales(
    #     modelos_por_grupo_agosto,
    #     X_predict_agosto,
    #     clientes_predict_agosto,
    #     df_predict_agosto,
    #     top_k=TOP_K,
    #     mes=FINAL_PREDIC_AGOSTO
    # )
    
    # # Guardar predicciones
    # guardar_predicciones_finales({"top_k": resultados_agosto["top_k_global"]}, f"{FINAL_PREDIC_AGOSTO}_global")
    # guardar_predicciones_finales({"top_k": resultados_agosto["top_k_grupos"]}, f"{FINAL_PREDIC_AGOSTO}_grupos")
    
    # # Guardar ganancias
    # resultados_agosto["ganancias"].to_csv(f"predict/ganancias_{STUDY_NAME}_{FINAL_PREDIC_AGOSTO}.csv", index=False)
    # logger.info(f"✅ CSV de ganancias guardado: predict/ganancias_{STUDY_NAME}_{FINAL_PREDIC_AGOSTO}.csv")

    # Resumen final
    logger.info("=== RESUMEN FINAL ===")
    logger.info("Entrenamiento final completado exitosamente")
    logger.info(f"Mejores hiperparámetros utilizados: {mejores_params}")
    logger.info(f"Log detallado: logs/{nombre_log}")
    logger.info(">>> Ejecución finalizada. Revisar logs para más detalles.")




    
    

if __name__ == "__main__":
    main()




## Fin del código main.py

## Codigo Muerto

    
        # Saco cpayroll_trx por tener mucho drifting
    
        # # 1.5 PSI para detectar data drifting
        # num_cols = df.select_dtypes(include=[np.number]).columns
        # psi_resultados = psi_by_columns(df,num_cols, 202104, 202106,"foto_mes")
        # psi_resultados.to_csv("feature_importance/psi_resultados.csv")
        # Saco cpayroll_trx por tener mucho drifting
        # df = df.drop(columns="cpayroll_trx")
    
    
        # df_to_select_columns = pd.read_csv("feature_importance/feature_importance_sin_canarios.csv").sort_values("importance",ascending=False)
    
        
        # Leer el archivo de importancias
        # df_columnas_poco_importantes = pd.read_csv("feature_importance/feature_importance_Retesting...Saco Enero y Febrero para Limpiar FI 0 Vars_2025-10-10_12-56-35.csv")
    
        # # Filtrar las features con importance_split <= 1
        # columnas_poco_importantes = df_columnas_poco_importantes.loc[
        #     df_columnas_poco_importantes['importance_split'] == 0, 
        #     'feature'
        # ].tolist()
    
        # Top 40 de features de mayor importancia
    
        # # columnas_40_mas_importantes = df_to_select_columns.head(40)["feature"].to_list()
        # columnas_40_mas_importantes = ["ctrx_quarter","mpayroll","cpayroll_trx","mprestamos_personales","mcuentas_saldo","mpasivos_margen","mcaja_ahorro","mtarjeta_visa_consumo","mrentabilidad_annual","Visa_msaldopesos","ctarjeta_visa_transacciones","cliente_edad","mactivos_margen","ctarjeta_master","Master_fechaalta","Visa_fechaalta","Visa_Fvencimiento","Visa_msaldototal","TC_Total_mpagospesos","TC_Total_mpagominimo","mtransferencias_recibidas","TC_Total_fechaalta","Master_Fvencimiento","numero_de_cliente","cliente_antiguedad","mrentabilidad","ctarjeta_debito_transacciones","TC_Total_msaldototal","chomebanking_transacciones","Visa_mpagospesos","ccomisiones_otras","Visa_mpagominimo","mcomisiones","mpayroll_corregida", "cpayroll_trx_corregida","ctrx_30d","ctrx_60d","saldo_total","uso_credito_ratio","TC_Total_msaldototal","uso_tarjeta_ratio","flujo_netotransf","uso_digital_ratio"]
    

        # df_fe = feature_engineering_ratio(df_fe,columnas_40_mas_importantes)
    
        # columnas_Master = [c for c in columnas_base if c.startswith("Master_")]
        # columnas_Visa = [c for c in columnas_base if c.startswith("Visa_")]      
        # columnas_categoricas = [c for c in columnas_base if df[c].nunique() < 5]

        # df_fe = df_fe.astype({col: "float32" for col in df_fe.select_dtypes("float").columns})


                # df_polars = pl.from_pandas(df_fe)  # si tu df original era Pandas

        # excluir = ["numero_de_cliente", "target", "foto_mes", "target_to_calculate_gan"]
        # columnas_a_normalizar = [c for c in df_polars.columns if c not in excluir and df_polars[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]

        # df_polars = feature_engineering_robust_by_month_polars(df_polars, columnas=columnas_a_normalizar)
        
        # # Si querés volver a Pandas
        # df_fe = df_polars.to_pandas()

    
        # df_fe = df_fe.astype({col: "float32" for col in df_fe.select_dtypes("float").columns})
        # df_fe = df_fe[[c for c in df_fe.columns if not re.search(r'_delta_\d+_delta_', c)]]
        # df_fe = df_fe[[c for c in df_fe.columns if not re.search(r'_delta_\d+_\d+$', c)]]
        # df_fe = df_fe[[c for c in df_fe.columns if not re.search(r'lag\d+lag', c)]]
        # df_fe = df_fe[[c for c in df_fe.columns if not re.search(r'lag\d+_\d+$', c)]]
    
        # df_fe = feature_engineering_variables_canarios(df_fe)
    
        # Eliminar las columnas poco importantes de df_fe
        # df_fe = df_fe.drop(columns=columnas_poco_importantes, errors='ignore')
    
        # logger.info(f"Se eliminaron {len(columnas_poco_importantes)} columnas con importance_split <= 1")



    # Evaluar en test
    # resultados_test, y_pred_proba, y_test = evaluar_en_test_ensamble(df_fe, mejores_params)
    
    # res = comparar_semillas_en_grafico(df_fe, mejores_params, SEMILLA, study_name=STUDY_NAME)

    # # Simular distribución de ganancias
    # ganancias_sim = muestrear_ganancias(y_test, y_pred_proba)
             
    # logger.info("=== GRAFICO DE TEST ===")

    # # Grafico de test
    # logger.info("=== GRAFICO DE TEST ===")
    # ruta_grafico = crear_grafico_ganancia_avanzado(y_test,y_pred_proba)


        # mejores_params = {'num_leaves': 23, 'lr_init': 0.14053552566659705, 'min_data_in_leaf': 223, 'feature_fraction': 0.6616669584635271, 'bagging_fraction': 0.23994377622330532, 'num_boost_round': 439, 'lr_decay': 0.9124750514032693}







        # # Excluyo variables con baja importancia
        # importance_df = pd.read_csv("../../../buckets/b1/Compe_02/data/feature_importance_to_remove_variables.csv")
        
        # # Limpiar nombres de columnas por si hay espacios
        # importance_df.columns = importance_df.columns.str.strip()
        
        # # Limpiar y convertir la columna 'importance_split' a numérica
        # importance_df['importance_split'] = (
        #     importance_df['importance_split']
        #     .astype(str)
        #     .str.replace(',', '.', regex=False)
        #     .astype(float)
        # )

        # # Identificar las variables con importance_split < 10
        # features_a_excluir = importance_df.loc[importance_df['importance_split'] < 10, 'feature'].tolist()
        
        # # Excluir esas columnas de df_fe (solo si existen)
        # df_fe = df_fe.drop(columns=[c for c in features_a_excluir if c in df_fe.columns], errors='ignore')
        
        # logger.info(f"Se excluyeron {len(features_a_excluir)} variables con importance_split < 10")







        # # Pruebo con variables de fintches

        # # Importar el archivo con tasas y penetración
        # fintechs_df = pd.read_csv("../../../buckets/b1/Compe_02/data/tasa_y_penetracion_mensual_argentina_2018_2025.csv")
        
        # # Limpiar nombres de columnas
        # fintechs_df.columns = fintechs_df.columns.str.strip()
        
        # # Renombrar columnas para facilitar el merge y evitar símbolos
        # fintechs_df = fintechs_df.rename(columns={
        #     'Fecha en formato foto_mes': 'foto_mes',
        #     'Tasa_interes_money_market_TNA_estimada_%': 'tasa_interes_mm_tna',
        #     'Penetracion_billeteras_%': 'penetracion_billeteras'
        # })
        
        # # Asegurar tipo de dato consistente con df_fe
        # fintechs_df['foto_mes'] = fintechs_df['foto_mes'].astype(int)
        
        # # Merge (left join) para agregar las variables a cada registro de df_fe según su foto_mes
        # df_fe = df_fe.merge(
        #     fintechs_df[['foto_mes', 'tasa_interes_mm_tna', 'penetracion_billeteras']],
        #     on='foto_mes',
        #     how='left'
        # )
        
        # logger.info("Se agregaron variables macroeconómicas (tasa_interes_mm_tna, penetracion_billeteras) según foto_mes.")



    # mejores_params = {'num_leaves': 169, 'learning_rate': 0.01653493811854045, 'min_data_in_leaf': 666, 'feature_fraction': 0.22865878320049338, 'bagging_fraction': 0.7317466615048293, 'num_boost_round': 682}



    # # 06 Entrenar modelo final (semillerio)
    # logger.info("=== ENTRENAMIENTO FINAL ===")
    # logger.info("Preparar datos para entrenamiento final")
    # X_train, y_train, X_predict, clientes_predict = preparar_datos_entrenamiento_final(df_fe)
  
    # # Entrenar modelo final
    # logger.info("Entrenar modelo final")
    # _ , modelo_final = entrenar_modelo_final_undersampling(X_train, y_train, X_predict ,mejores_params, SEMILLA, ratio_undersampling = 1)

  
    # # Generar predicciones finales
    # logger.info("Generar predicciones finales")
    # resultados = generar_predicciones_finales(modelo_final, X_predict, clientes_predict, umbral=UMBRAL, top_k=TOP_K)


    # # Guardar predicciones
    # logger.info("Guardar predicciones")
    # archivo_salida = guardar_predicciones_finales(resultados)



    # mejores_params = {'bagging_fraction': 0.9366158838759591, 'feature_fraction': 0.6097465146850822, 'lambda_l1': 1.8715916172393408, 'lambda_l2': 0.47499514072885834, 'learning_rate': 0.03421069355219755, 'min_data_in_leaf': 19, 'num_boost_round': 1562, 'num_leaves': 151}

