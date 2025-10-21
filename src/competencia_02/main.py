import logging
from datetime import datetime
import os
import pandas as pd
import numpy as np
from features import feature_engineering_lag, feature_engineering_delta, feature_engineering_regr_slope_window, feature_engineering_ratio, feature_engineering_tc_total, generar_ctrx_features, feature_engineering_cpayroll_trx_corregida, feature_engineering_mpayroll_corregida, variables_aux
from loader import cargar_datos, convertir_clase_ternaria_a_target
from optimization import *
from best_params import cargar_mejores_hiperparametros
from final_training import preparar_datos_entrenamiento_final, generar_predicciones_finales, entrenar_modelo_final,feature_importance
from output_manager import guardar_predicciones_finales
from best_params import obtener_estadisticas_optuna
from config import *
from test import *
from grafico_test import *
import re

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
    df = cargar_datos("../../../datasets/competencia_01_crudo.csv")
    if df is None:
        logger.error("No se pudieron cargar los datos; 'cargar_datos' retornó None.")
        raise ValueError("cargar_datos devolvió None. Verificar ruta o contenido de 'data/competencia_01_crudo.csv'.")
    logger.info(f"Datos cargados: {df.shape}")


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


    


    # 2. Feature Engineering
    # Excluyo las variables no corregidas

    # df_fe = feature_engineering_cpayroll_trx_corregida(df)
    # df_fe = feature_engineering_mpayroll_corregida(df_fe)
    df_fe = feature_engineering_tc_total(df_fe)
    df_fe = generar_ctrx_features(df_fe)
    df_fe = variables_aux(df_fe)
    columnas_base = df_fe.columns.tolist()
    columnas_a_excluir = ["foto_mes","cliente_edad","numero_de_cliente","target","target_to_calculate_gan"]
    atributos = [c for c in columnas_base if c not in columnas_a_excluir]
    # df_fe = feature_engineering_ratio(df_fe,columnas_40_mas_importantes)

    # columnas_Master = [c for c in columnas_base if c.startswith("Master_")]
    # columnas_Visa = [c for c in columnas_base if c.startswith("Visa_")]      
    # columnas_categoricas = [c for c in columnas_base if df[c].nunique() < 5]
    for i in (1,2):
        df_fe = feature_engineering_lag(df_fe, columnas=atributos, cant_lag=i)
    for i in (1,2):
        df_fe = feature_engineering_delta(df_fe, columnas=atributos, cant_delta=i)
    df_fe = feature_engineering_delta(df_fe, columnas=atributos, cant_delta=2)
    df_fe = df_fe.astype({col: "float32" for col in df_fe.select_dtypes("float").columns})
    for i in (2,3):
        df_fe = feature_engineering_regr_slope_window(df_fe, columnas=atributos, ventana = i)

    # montos_vars = ["matm", "matm_other", "mautoservicio", "mcaja_ahorro", "mcaja_ahorro_adicional", "mcaja_ahorro_dolares", "mcajeros_propios_descuentos", "mcheques_depositados", "mcheques_depositados_rechazados", "mcheques_emitidos", "mcheques_emitidos_rechazados", "mcomisiones", "mcomisiones_mantenimiento", "mcomisiones_otras", "mcuenta_corriente", "mcuenta_corriente_adicional", "mcuenta_debitos_automaticos", "mcuentas_saldo", "mextraccion_autoservicio", "mforex_buy", "mforex_sell", "minversion1_dolares", "minversion1_pesos", "minversion2", "mora_total", "mpagodeservicios", "mpagomiscuentas", "mpasivos_margen", "mpayroll", "mpayroll_corregida", "mpayroll2", "mplazo_fijo_dolares", "mplazo_fijo_pesos", "mprestamos_hipotecarios", "mprestamos_personales", "mprestamos_prendarios", "mrentabilidad", "mrentabilidad_annual", "mtarjeta_master_consumo", "mtarjeta_master_descuentos", "mtarjeta_visa_consumo", "mtarjeta_visa_descuentos", "mtransferencias_emitidas", "mtransferencias_recibidas", "mttarjeta_master_debitos_automaticos", "mttarjeta_visa_debitos_automaticos", "mactivos_margen", "margen_por_cuenta", "margen_por_producto", "margen_total", "Master_madelantodolares", "Master_madelantopesos", "Master_mconsumosdolares", "Master_mconsumospesos", "Master_mconsumototal", "Master_mfinanciacion_limite", "Master_mlimitecompra", "Master_mpagado", "Master_mpagominimo", "Master_mpagosdolares", "Master_mpagospesos", "Master_msaldodolares", "Master_msaldopesos", "Master_msaldototal", "Visa_madelantodolares", "Visa_madelantopesos", "Visa_mconsumosdolares", "Visa_mconsumospesos", "Visa_mconsumototal", "Visa_mfinanciacion_limite", "Visa_mlimitecompra", "Visa_mpagado", "Visa_mpagominimo", "Visa_mpagosdolares", "Visa_mpagospesos", "Visa_msaldodolares", "Visa_msaldopesos", "Visa_msaldototal", "TC_Total_madelantodolares", "TC_Total_madelantopesos", "TC_Total_mconsumosdolares", "TC_Total_mconsumospesos", "TC_Total_mconsumototal", "TC_Total_mfinanciacion_limite", "TC_Total_mlimitecompra", "TC_Total_mpagado", "TC_Total_mpagominimo", "TC_Total_mpagosdolares", "TC_Total_mpagospesos", "TC_Total_msaldodolares", "TC_Total_msaldopesos", "TC_Total_msaldototal"]



    # df_fe = feature_engineering_regr_slope_window(df_fe, columnas=atributos, ventana = 2)

    variables_con_drfting =["Visa_Finiciomora","Master_fultimo_cierre","Visa_fultimo_cierre","Master_Finiciomora","cpayroll_trx","mpayroll"]


    df_fe = df_fe.drop(columns=variables_con_drfting, errors='ignore')



    # df_fe = df_fe.astype({col: "float32" for col in df_fe.select_dtypes("float").columns})
    # df_fe = df_fe[[c for c in df_fe.columns if not re.search(r'_delta_\d+_delta_', c)]]
    # df_fe = df_fe[[c for c in df_fe.columns if not re.search(r'_delta_\d+_\d+$', c)]]
    # df_fe = df_fe[[c for c in df_fe.columns if not re.search(r'lag\d+lag', c)]]
    # df_fe = df_fe[[c for c in df_fe.columns if not re.search(r'lag\d+_\d+$', c)]]

    # df_fe = feature_engineering_variables_canarios(df_fe)

    # Eliminar las columnas poco importantes de df_fe
    # df_fe = df_fe.drop(columns=columnas_poco_importantes, errors='ignore')

    # logger.info(f"Se eliminaron {len(columnas_poco_importantes)} columnas con importance_split <= 1")
    
    logger.info(f"Feature Engineering completado: {df_fe.shape}")
    

    # 3. Convertir clase_ternaria a binario
    df_fe = convertir_clase_ternaria_a_target(df_fe)

    # df_fe = df_fe.astype({col: "float32" for col in df_fe.select_dtypes("float").columns})

    # df_fe.to_csv("data/competencia_fe_.csv", index=False)


    # 4. Ejecutar optimización (función simple)
    study = optimizar(df_fe, n_trials=100, undersampling = 0.2)
  
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
  
    # logger.info("=== OPTIMIZACIÓN COMPLETADA ===")

     #05 Test en mes desconocido
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    # Cargar mejores hiperparámetros

    # mejores_params = {'num_leaves': 46, 'learning_rate': 0.016377657023274192, 'min_data_in_leaf': 710, 'feature_fraction': 0.2503218637353462, 'bagging_fraction': 0.2352773905721117, 'num_boost_round': 1000}

    # mejores_params = {
    #   "num_leaves": 480,
    #   "learning_rate": 0.0317163548889023,
    #   "min_data_in_leaf": 177,
    #   "feature_fraction": 0.5662761244610663,
    #   "bagging_fraction": 0.21270355026382087,
    #   "min_gain_to_split": 0.038636072006724934,
    #   "num_boost_round": 1062
    # }

    # mejores_params = {'num_leaves': 173, 'learning_rate': 0.05341811211926391, 'min_data_in_leaf': 751, 'feature_fraction': 0.28618267208195725, 'bagging_fraction': 0.380548133981898, 'num_boost_round': 824}

    mejores_params = cargar_mejores_hiperparametros()

  
    # Evaluar en test
    resultados_test, y_pred_proba, y_test = evaluar_en_test(df_fe, mejores_params)
    
    res = comparar_semillas_en_grafico(df_fe, mejores_params, SEMILLA, study_name=STUDY_NAME)

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

