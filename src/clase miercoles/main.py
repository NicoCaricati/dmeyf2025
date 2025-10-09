import logging
from datetime import datetime
import os
import pandas as pd
import numpy as np
from features import feature_engineering_lag, feature_engineering_delta, feature_engineering_regr_slope_window, feature_engineering_ratio, feature_engineering_variables_canarios, feature_engineering_tc_total, generar_ctrx_features, calculate_psi, psi_by_columns
from loader import cargar_datos, convertir_clase_ternaria_a_target
from optimization import *
from best_params import cargar_mejores_hiperparametros
from final_training import preparar_datos_entrenamiento_final, generar_predicciones_finales, entrenar_modelo_final,feature_importance
from output_manager import guardar_predicciones_finales
from best_params import obtener_estadisticas_optuna
from config import *
from test import *
from grafico_test import *

### Configuración de logging ###
os.makedirs("logs", exist_ok=True)
fecha = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
nombre_log = f"log_{STUDY_NAME}_{fecha}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler("logs/" + nombre_log),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Iniciando programa de optimización con log fechado")

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


### Main ###
def main():
    """Pipeline principal con optimización usando configuración YAML."""
    logger.info("=== INICIANDO OPTIMIZACIÓN CON CONFIGURACIÓN YAML ===")
  
    # 1. Cargar datos
    df = cargar_datos("data/competencia_01_crudo.csv")
    logger.info(f"Datos cargados: {df.shape}")

    # Saco cpayroll_trx por tener mucho drifting

    # # 1.5 PSI para detectar data drifting
    # num_cols = df.select_dtypes(include=[np.number]).columns
    # psi_resultados = psi_by_columns(df,num_cols, 202104, 202106,"foto_mes")
    # psi_resultados.to_csv("feature_importance/psi_resultados.csv")
    # Saco cpayroll_trx por tener mucho drifting
    # df = df.drop(columns="cpayroll_trx")


    df_to_select_columns = pd.read_csv("feature_importance/feature_importance_sin_canarios.csv").sort_values("importance",ascending=False)
    
    # Leer el archivo de importancias
    # df_columnas_poco_importantes = pd.read_csv("feature_importance/feature_importance_Going Back to BM - Removing 0 FE Vars.csv")

    # Filtrar las features con importance_split <= 1
    # columnas_poco_importantes = df_columnas_poco_importantes.loc[
    #     df_columnas_poco_importantes['importance_split'] == 0, 
    #     'feature'
    # ].tolist()

    # Top 40 de features de mayor importancia

    columnas_40_mas_importantes = df_to_select_columns.head(40)["feature"].to_list()

    # 2. Feature Engineering
    df_fe = feature_engineering_tc_total(df)
    df_fe = generar_ctrx_features(df_fe)
    df_fe = feature_engineering_ratio(df_fe,columnas_40_mas_importantes)
    columnas_a_excluir = ["foto_mes","cliente_edad","numero_de_cliente","target"]
    atributos = [c for c in df.columns if c not in columnas_a_excluir]
    columnas_Master = [c for c in df.columns if c.startswith("Master_")]
    columnas_Visa = [c for c in df.columns if c.startswith("Visa_")]      
    columnas_categoricas = [c for c in df.columns if df[c].nunique() < 5]
    for i in (1,2):
        df_fe = feature_engineering_lag(df_fe, columnas=atributos, cant_lag=i)
    for i in (1,2):
        df_fe = feature_engineering_delta(df_fe, columnas=atributos, cant_delta=i)
    
    df_fe = feature_engineering_regr_slope_window(df_fe, columnas=atributos, ventana = 3)

    # df_fe = feature_engineering_variables_canarios(df_fe)

    # Eliminar las columnas poco importantes de df_fe
    # df_fe = df_fe.drop(columns=columnas_poco_importantes, errors='ignore')

    # logger.info(f"Se eliminaron {len(columnas_poco_importantes)} columnas con importance_split <= 1")
    
    logger.info(f"Feature Engineering completado: {df_fe.shape}")
    

    # 3. Convertir clase_ternaria a binario
    df_fe = convertir_clase_ternaria_a_target(df_fe)

    # df_fe = df_fe.astype({col: "float32" for col in df_fe.select_dtypes("float").columns})

    # df_fe.to_csv("data/competencia_fe_.csv", index=False)

    # # # 3.5 Muestreo para acelerar optimización (opcional)
    # # clientes_202101 = df_fe[df_fe['foto_mes'] == 202101]['numero_de_cliente'].unique()
    # # clientes_muestra = np.random.choice(clientes_202101, size=int(0.45 * len(clientes_202101)), replace=False)
    # # df_fe_sampled = df_fe[df_fe['numero_de_cliente'].isin(clientes_muestra)]
    # logger.info(f"Datos muestreados para optimización: {df_fe_sampled.shape}")
    # saco los meses 5 y 6 para que no haya fugas
    df_fe_sampled = df_fe
    df_fe_sampled = df_fe_sampled[~df_fe_sampled['foto_mes'].isin([202105,202106])]
    logger.info(f"Excluyo de la muestra meses 5 y 6")
    # df_fe_sampled.to_csv("data/competencia_fe_sampled.csv", index=False)

    # 4. Ejecutar optimización (función simple)
    study = optimizar_con_cv(df_fe_sampled, n_trials=100)
  
    # 5. Análisis adicional
    logger.info("=== ANÁLISIS DE RESULTADOS ===")
    trials_df = study.trials_dataframe()
    if len(trials_df) > 0:
        top_5 = trials_df.nlargest(5, 'value')
        logger.info("Top 5 mejores trials:")
        for idx, trial in top_5.iterrows():
            logger.info(
            f"Trial {int(trial['number'])}: "
            f"Ganancia = {trial['value']:,.0f} | "
            f"Parámetros: {trial['params']}")
  
    logger.info("=== OPTIMIZACIÓN COMPLETADA ===")

     #05 Test en mes desconocido
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    # Cargar mejores hiperparámetros
    # mejores_params = {'num_leaves': 46, 'learning_rate': 0.016377657023274192, 'min_data_in_leaf': 710, 'feature_fraction': 0.2503218637353462, 'bagging_fraction': 0.2352773905721117}
    # mejores_params =  {
    #   "num_leaves": 468,
    #   "learning_rate": 0.09008244016707123,
    #   "min_data_in_leaf": 238,
    #   "feature_fraction": 0.5481791814620421,
    #   "bagging_fraction": 0.7807387893805953,
    #   "min_gain_to_split": 0.0014506741831737321,
    #   "num_boost_round": 1061
    # }

    # mejores_params = {'num_leaves': 46, 'learning_rate': 0.016377657023274192, 'min_data_in_leaf': 710, 'feature_fraction': 0.2503218637353462, 'bagging_fraction': 0.2352773905721117, 'num_boost_round': 1000}

    mejores_params = cargar_mejores_hiperparametros()

  
    # Evaluar en test
    resultados_test, y_pred_proba, y_test = evaluar_en_test(df_fe, mejores_params)
    
    # resultados_test_v2 = evaluar_en_test_v2(df_fe, mejores_params)

    # Simular distribución de ganancias
    ganancias_sim = muestrear_ganancias(y_test, y_pred_proba)
  
    # Guardar resultados de test
    guardar_resultados_test(resultados_test)
  
    # Resumen de evaluación en test
    logger.info("=== RESUMEN DE EVALUACIÓN EN TEST ===")
    logger.info(f"✅ Ganancia en test: {resultados_test['ganancia_test']:,.0f}")
    logger.info(f"🎯 Predicciones positivas: {resultados_test['predicciones_positivas']:,} ({resultados_test['porcentaje_positivas']:.2f}%)")

 
    logger.info("=== GRAFICO DE TEST ===")

    # Graficar y guardar
    graficar_distribucion_ganancia(ganancias_sim, modelo_nombre= STUDY_NAME)

    # Registrar resultados en CSV comparativo
    registrar_resultados_modelo(STUDY_NAME, ganancias_sim)

    # Grafico de test
    logger.info("=== GRAFICO DE TEST ===")
    ruta_grafico = crear_grafico_ganancia_avanzado(y_test,y_pred_proba)


    #06 Entrenar modelo final
    logger.info("=== ENTRENAMIENTO FINAL ===")
    logger.info("Preparar datos para entrenamiento final")
    X_train, y_train, X_predict, clientes_predict = preparar_datos_entrenamiento_final(df_fe)
  
    # Entrenar modelo final
    logger.info("Entrenar modelo final")
    modelo_final = entrenar_modelo_final(X_train, y_train, mejores_params)
  
    # Generar predicciones finales
    logger.info("Generar predicciones finales")
    resultados = generar_predicciones_finales(modelo_final, X_predict, clientes_predict, umbral=UMBRAL, top_k=TOP_K)
  
    # Guardar predicciones
    logger.info("Guardar predicciones")
    archivo_salida = guardar_predicciones_finales(resultados)
  
    # Resumen final
    logger.info("=== RESUMEN FINAL ===")
    logger.info(f"Entrenamiento final completado exitosamente")
    logger.info(f"Mejores hiperparámetros utilizados: {mejores_params}")
    logger.info(f"Períodos de entrenamiento: {FINAL_TRAIN}")
    logger.info(f"Período de predicción: {FINAL_PREDIC}")
    logger.info(f"Archivo de salida: {archivo_salida}")
    logger.info(f"Log detallado: logs/{nombre_log}")


    logger.info(f">>> Ejecución finalizada. Revisar logs para mas detalles.")

if __name__ == "__main__":
    main()

