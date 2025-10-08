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


    #df_to_select_columns = pd.read_csv("feature_importance/feature_importance_sin_canarios.csv").sort_values("importance",ascending=False)
    # Leer el archivo de importancias
    #df_columnas_poco_importantes = pd.read_csv("feature_importance/feature_importance_Going Back to BM - Removing 0 FE Vars.csv")

    # Filtrar las features con importance_split <= 1
    columnas_poco_importantes = ["mprestamos_prendarios_delta_1_1","cprestamos_prendarios_delta_2","cprestamos_prendarios_delta_1_1","ccheques_depositados_delta_1_1","mforex_buy_delta_2","mforex_buy_delta_1_1","cforex_buy_delta_2","cliente_antiguedad_delta_2","cliente_antiguedad_delta_1_1","cliente_vip_delta_2","mcaja_ahorro_adicional_delta_2","ccaja_ahorro_delta_2","ccaja_ahorro_delta_1_1","mcuenta_corriente_adicional_delta_2","mcuenta_corriente_adicional_delta_1_1","ccuenta_corriente_delta_2","ccuenta_corriente_delta_1_1","tcuentas_delta_2","cliente_vip_delta_1_1","TC_Total_Fvencimiento_delta_1","TC_Total_Finiciomora_delta_1","TC_Total_fechaalta_delta_1","ccheques_emitidos_rechazados_delta_1_1","mcheques_depositados_rechazados_delta_2","mcheques_depositados_rechazados_delta_1_1","ccheques_depositados_rechazados_delta_2","ccheques_depositados_rechazados_delta_1_1","mcheques_emitidos_delta_1_1","ccheques_emitidos_delta_2","ccheques_emitidos_delta_1_1","mcheques_depositados_delta_2","ccheques_depositados_delta_2","Master_madelantodolares_delta_1","Master_madelantopesos_delta_1","Master_mconsumosdolares_delta_1","Master_msaldodolares_delta_1","mcheques_emitidos_rechazados_delta_2","mcheques_emitidos_rechazados_delta_1_1","ccheques_emitidos_rechazados_delta_2","Visa_madelantodolares_delta_1","Visa_madelantopesos_delta_1","Visa_delinquency_delta_1","TC_Total_madelantopesos_delta_1","TC_Total_madelantodolares_delta_1","TC_Total_cadelantosefectivo_delta_1","Visa_cadelantosefectivo_delta_1","Visa_fechaalta_delta_1","ccheques_emitidos_rechazados_delta_1","mcheques_depositados_rechazados_delta_1","ccheques_depositados_rechazados_delta_1","ccheques_depositados_delta_1","Master_Finiciomora_delta_1","Master_cadelantosefectivo_delta_1","Master_mpagosdolares_delta_1","mpagodeservicios_delta_1","cpagodeservicios_delta_1","cpayroll2_trx_delta_1","mpayroll2_delta_1","Master_Finiciomora_delta_2","Master_Finiciomora_delta_1_1","Master_Fvencimiento_delta_1_1","Master_status_delta_1_1","Master_delinquency_delta_2","Master_delinquency_delta_1_1","mforex_buy_delta_1","cforex_buy_delta_1","mcheques_emitidos_rechazados_delta_1","cinversion2_delta_1","minversion1_dolares_delta_1","cinversion1_delta_1","mplazo_fijo_pesos_delta_1","cplazo_fijo_delta_1","cprestamos_hipotecarios_delta_1","mprestamos_prendarios_delta_1","Master_madelantodolares_delta_2","Master_madelantodolares_delta_1_1","Master_madelantopesos_delta_2","Master_madelantopesos_delta_1_1","Master_mconsumosdolares_delta_2","Master_mconsumosdolares_delta_1_1","Master_msaldodolares_delta_1_1","mcajeros_propios_descuentos_delta_1","cprestamos_personales_delta_1","ctarjeta_master_delta_1","ctarjeta_debito_delta_1","cdescubierto_preacordado_delta_1","Master_cadelantosefectivo_delta_2","Master_cadelantosefectivo_delta_1_1","Master_fechaalta_delta_2","Master_fechaalta_delta_1_1","Master_mpagosdolares_delta_2","ccaja_seguridad_delta_1","cseguro_accidentes_personales_delta_1","cseguro_vivienda_delta_1","cseguro_auto_delta_1","cseguro_vida_delta_1","minversion2_delta_1","TC_Total_Finiciomora_lag_2","TC_Total_status_lag_3","TC_Total_status_lag_2","TC_Total_delinquency_lag_3","Visa_Finiciomora_delta_2","Visa_Finiciomora_delta_1_1","Visa_Fvencimiento_delta_2","ccaja_ahorro_delta_1","mcuenta_corriente_adicional_delta_1","ccuenta_corriente_delta_1","tcuentas_delta_1","cliente_antiguedad_delta_1","internet_delta_1","cliente_vip_delta_1","cprestamos_prendarios_delta_1","TC_Total_madelantopesos_lag_3","TC_Total_madelantopesos_lag_2","TC_Total_madelantopesos_lag_1_1","TC_Total_madelantodolares_lag_3","TC_Total_madelantodolares_lag_2","TC_Total_madelantodolares_lag_1_1","TC_Total_cadelantosefectivo_lag_3","Visa_madelantodolares_delta_2","Visa_madelantodolares_delta_1_1","Visa_madelantopesos_delta_2","Visa_madelantopesos_delta_1_1","TC_Total_mpagosdolares_lag_1_1","TC_Total_delinquency_lag_2","TC_Total_delinquency_lag_1_1","TC_Total_msaldodolares_lag_3","TC_Total_Finiciomora_lag_3","Visa_madelantodolares_lag_3","Visa_madelantodolares_lag_2","Visa_madelantodolares_lag_1_1","TC_Total_madelantodolares_delta_1_1","TC_Total_cadelantosefectivo_delta_2","TC_Total_cadelantosefectivo_delta_1_1","Visa_cadelantosefectivo_delta_2","Visa_cadelantosefectivo_delta_1_1","Visa_fechaalta_delta_2","Visa_fechaalta_delta_1_1","TC_Total_cadelantosefectivo_lag_2","TC_Total_cadelantosefectivo_lag_1_1","Visa_cadelantosefectivo_lag_3","Visa_cadelantosefectivo_lag_2","Visa_cadelantosefectivo_lag_1_1","Master_cadelantosefectivo_lag_3","Master_cadelantosefectivo_lag_2","TC_Total_madelantopesos_delta_2","TC_Total_madelantopesos_delta_1_1","TC_Total_madelantodolares_delta_2","Visa_Finiciomora_lag_2","Visa_status_lag_3","Visa_status_lag_2","Visa_madelantopesos_lag_3","Visa_madelantopesos_lag_2","Visa_madelantopesos_lag_1_1","Visa_msaldodolares_lag_3","Visa_mpagosdolares_lag_3","Master_mconsumosdolares_lag_1_1","Master_msaldodolares_lag_3","TC_Total_fechaalta_delta_1_1","TC_Total_status_delta_2","TC_Total_status_delta_1_1","TC_Total_delinquency_delta_1_1","TC_Total_mpagosdolares_delta_2","Master_mpagosdolares_lag_3","Master_mpagosdolares_lag_2","Master_mpagado_lag_3","Master_madelantodolares_lag_3","Visa_status_lag_1_1","Visa_delinquency_lag_2","Visa_delinquency_lag_1_1","tmobile_app_lag_2","tmobile_app_lag_1_1","slope_cliente_vip_window","TC_Total_fultimo_cierre_delta_2","TC_Total_Finiciomora_delta_2","TC_Total_Finiciomora_delta_1_1","TC_Total_fechaalta_delta_2","Master_msaldodolares_lag_1_1","Master_Finiciomora_lag_3","Master_Finiciomora_lag_2","Master_madelantodolares_lag_2","Master_madelantodolares_lag_1_1","Master_madelantopesos_lag_3","Master_madelantopesos_lag_2","Master_madelantopesos_lag_1_1","Master_mconsumosdolares_lag_3","ccajas_transacciones_lag_3","slope_mcuenta_corriente_adicional_window","ccajas_otras_lag_3","ccajas_otras_lag_2","ccajas_extracciones_lag_2","Master_status_lag_2","Master_delinquency_lag_2","Master_delinquency_lag_1_1","tmobile_app_lag_3","mcheques_emitidos_rechazados_lag_1_1","ccheques_emitidos_rechazados_lag_3","ccheques_emitidos_rechazados_lag_2","ccheques_emitidos_rechazados_lag_1_1","mcheques_depositados_rechazados_lag_3","mcheques_depositados_rechazados_lag_2","mcheques_depositados_rechazados_lag_1_1","ccheques_depositados_rechazados_lag_3","slope_minversion1_dolares_window","slope_cinversion1_window","slope_mplazo_fijo_pesos_window","Master_cadelantosefectivo_lag_1_1","slope_mprestamos_prendarios_window","ccajas_depositos_lag_2","ccajas_consultas_lag_3","mcheques_depositados_rechazados","ccheques_emitidos_lag_2","ccheques_emitidos_lag_1_1","cplazo_fijo_delta_2","mcheques_depositados_lag_2","mpagodeservicios","ccheques_depositados_lag_3","ccheques_depositados_lag_2","ccheques_depositados_lag_1_1","slope_cpayroll2_trx_window","slope_mpayroll2_window","cseguro_vida_delta_1_1","foto_mes","tcallcenter_lag_2","tcallcenter_lag_1_1","mcheques_emitidos_rechazados_lag_3","mcheques_emitidos_rechazados_lag_2","cforex_buy_lag_3","cforex_buy_lag_2","cforex_buy_lag_1_1","active_quarter","cforex_lag_2","cforex_lag_1_1","cinversion2_delta_2","slope_mtarjeta_master_descuentos_window","slope_mpagodeservicios_window","slope_cpagodeservicios_window","ccheques_depositados_rechazados_lag_2","ccheques_depositados_rechazados_lag_1_1","mcheques_emitidos_lag_3","cprestamos_hipotecarios_delta_2","mcheques_emitidos_lag_1_1","ccheques_emitidos_lag_3","ctarjeta_master_descuentos_lag_3","ctarjeta_master_descuentos_lag_2","slope_mcheques_depositados_rechazados_window","slope_ccheques_depositados_rechazados_window","slope_cforex_sell_window","slope_mforex_buy_window","slope_cforex_buy_window","mforex_sell_lag_3","mforex_sell_lag_2","cpayroll2_trx_delta_2","cforex_sell_lag_3","cforex_sell_lag_2","cforex_sell_lag_1_1","mforex_buy_lag_3","mforex_buy_lag_2","mforex_buy_lag_1_1","cpagodeservicios_lag_3","cpagodeservicios_lag_2","cpagodeservicios_lag_1_1","cpayroll2_trx_delta_1_1","ctarjeta_master_debitos_automaticos_lag_3","slope_mcheques_emitidos_rechazados_window","slope_ccheques_emitidos_rechazados_window","ctarjeta_master_descuentos_lag_1_1","mtarjeta_visa_descuentos_lag_3","mpayroll2_delta_2","ctarjeta_visa_descuentos_lag_3","ctarjeta_visa_descuentos_lag_1_1","mcajeros_propios_descuentos_lag_1_1","ccajeros_propios_descuentos_lag_3","mpayroll2_delta_1_1","mtarjeta_master_descuentos_lag_2","cseguro_vivienda_lag_2","ccaja_seguridad_delta_2","slope_Master_madelantodolares_window","slope_Master_madelantopesos_window","slope_mprestamos_hipotecarios_window","ccaja_seguridad_delta_1_1","ccuenta_debitos_automaticos_lag_2","cseguro_accidentes_personales_delta_1_1","cpayroll2_trx_lag_3","cpayroll2_trx_lag_2","cpayroll2_trx_lag_1_1","mpayroll2_lag_3","mpayroll2_lag_2","mpagodeservicios_lag_3","mpagodeservicios_lag_2","mpagodeservicios_lag_1_1","cseguro_vida_lag_1_1","minversion2_lag_3","cseguro_vivienda_delta_2","cinversion2_lag_3","cinversion2_lag_2","minversion1_dolares_lag_3","minversion1_dolares_lag_2","minversion1_dolares_lag_1_1","slope_Master_cadelantosefectivo_window","mpayroll2_lag_1_1","cpayroll_trx_lag_1_1","ccaja_seguridad_lag_3","ccaja_seguridad_lag_1_1","cseguro_accidentes_personales_lag_3","cseguro_accidentes_personales_lag_2","cseguro_accidentes_personales_lag_1_1","cinversion1_lag_1_1","mplazo_fijo_pesos_lag_3","mplazo_fijo_pesos_lag_2","mplazo_fijo_pesos_lag_1_1","cplazo_fijo_lag_3","cplazo_fijo_lag_2","cplazo_fijo_lag_1_1","mprestamos_hipotecarios_lag_3","slope_TC_Total_cadelantosefectivo_window","slope_Visa_cadelantosefectivo_window","slope_Visa_madelantodolares_window","slope_Visa_madelantopesos_window","cseguro_vivienda_delta_1_1","cseguro_auto_lag_1_1","cseguro_vida_lag_3","cseguro_auto_delta_2","cseguro_auto_delta_1_1","ctarjeta_master_lag_1_1","ccuenta_corriente","slope_TC_Total_madelantopesos_window","slope_TC_Total_madelantodolares_window","mprestamos_hipotecarios_lag_1_1","cprestamos_hipotecarios_lag_3","cprestamos_hipotecarios_lag_2","cprestamos_hipotecarios_lag_1_1","mprestamos_prendarios_lag_3","cprestamos_prendarios_lag_3","cprestamos_prendarios_lag_2","cprestamos_personales_lag_1_1","minversion1_pesos_lag_2","cinversion1_lag_3","cinversion1_lag_2","mcuenta_corriente_adicional_lag_2","mcuenta_corriente_adicional_lag_1_1","ccuenta_corriente_lag_3","ccuenta_corriente_lag_2","ccuenta_corriente_lag_1_1","tcuentas_lag_3","ctarjeta_master_delta_2","slope_ccuenta_corriente_window_1","ctarjeta_master_delta_1_1","slope_cliente_vip_window_1","slope_TC_Total_Finiciomora_window","cdescubierto_preacordado_lag_1_1","ctarjeta_visa_lag_1_1","mcajeros_propios_descuentos_delta_2","cdescubierto_preacordado_lag_2","ctarjeta_master_transacciones_lag_2","TC_Total_madelantodolares_lag_1","TC_Total_cadelantosefectivo_lag_1","Visa_cadelantosefectivo_lag_1","slope_ctarjeta_master_window_1","ccajeros_propios_descuentos_delta_2","slope_mcuenta_corriente_adicional_window_1","cliente_vip_lag_3","cliente_vip_lag_2","cliente_vip_lag_1_1","mpagodeservicios_delta_2","active_quarter_lag_2","active_quarter_lag_1_1","TC_Total_Finiciomora_lag_1","TC_Total_status_lag_1","TC_Total_delinquency_lag_1","mcuenta_corriente_adicional_lag_3","mpagodeservicios_delta_1_1","cpagodeservicios_delta_2","slope_minversion1_dolares_window_1","slope_mplazo_fijo_pesos_window_1","slope_mprestamos_hipotecarios_window_1","slope_cprestamos_hipotecarios_window_1","slope_cprestamos_prendarios_window_1","Visa_delinquency_lag_1","Master_cadelantosefectivo_lag_1","Master_madelantodolares_lag_1","cpagodeservicios_delta_1_1","Visa_mpagosdolares_lag_1","cinversion2_delta_1_1","Visa_madelantodolares_lag_1","Visa_madelantopesos_lag_1","TC_Total_madelantopesos_lag_1","minversion1_dolares_delta_2","tcallcenter_lag_1","mcheques_emitidos_rechazados_lag_1","ccheques_emitidos_rechazados_lag_1","mcheques_depositados_rechazados_lag_1","ccheques_depositados_rechazados_lag_1","minversion1_dolares_delta_1_1","ccheques_emitidos_lag_1","minversion1_pesos_delta_2","slope_mpagodeservicios_window_1","slope_cpayroll2_trx_window_1","slope_mpayroll2_window_1","slope_cseguro_auto_window_1","Master_msaldodolares_lag_1","cinversion1_delta_2","Master_delinquency_lag_1","mplazo_fijo_pesos_lag_1","slope_mforex_buy_window_1","slope_cforex_buy_window_1","ctarjeta_master_descuentos_lag_1","mtarjeta_visa_descuentos_lag_1","mcajeros_propios_descuentos_lag_1","ccajeros_propios_descuentos_lag_1","mpagodeservicios_lag_1","cpagodeservicios_lag_1","ccheques_depositados_lag_1","cforex_sell_lag_1","mforex_buy_lag_1","cforex_buy_lag_1","cforex_lag_1","ccajas_otras_lag_1","ccajas_extracciones_lag_1","mprestamos_hipotecarios_lag_1","cprestamos_hipotecarios_lag_1","ctarjeta_master_lag_1","ctarjeta_visa_lag_1","slope_mcheques_emitidos_rechazados_window_1","slope_ccheques_emitidos_rechazados_window_1","cdescubierto_preacordado_lag_3","cinversion1_delta_1_1","cpayroll2_trx_lag_1","mpayroll2_lag_1","cseguro_accidentes_personales_lag_1","cseguro_auto_lag_1","mplazo_fijo_pesos_delta_2","cinversion2_lag_1","minversion1_dolares_lag_1","cinversion1_lag_1","TC_Total_madelantopesos","TC_Total_madelantodolares","TC_Total_cadelantosefectivo","Visa_cadelantosefectivo","TC_Total_delinquency","mplazo_fijo_pesos_delta_1_1","tcuentas_delta_1_1","cforex_buy_delta_1_1","ratio_cpayroll_trx_TC_Total_mpagominimo","ratio_mprestamos_personales_TC_Total_mpagominimo","cliente_vip_lag_1","active_quarter_lag_1","mcuenta_corriente_adicional_lag_1","ccuenta_corriente_lag_1","cforex_delta_2","cplazo_fijo_lag_1","ccheques_depositados_rechazados","cforex_delta_1_1","ccheques_depositados","mtarjeta_master_descuentos_delta_2","slope_Master_madelantodolares_window_1","slope_Master_madelantopesos_window_1","Master_delinquency","ccajas_depositos","ctarjeta_visa_delta_2","Master_madelantodolares","Master_madelantopesos","Master_Finiciomora","Visa_madelantodolares","Visa_madelantopesos","Visa_delinquency","Master_cadelantosefectivo","cliente_vip","cpagodeservicios","cpayroll2_trx","mpayroll2","ctarjeta_visa_delta_1_1","slope_Master_cadelantosefectivo_window_1","mforex_sell","cforex_sell","mforex_buy","cforex_buy","ctarjeta_debito_delta_1_1","ctarjeta_visa_descuentos","mcajeros_propios_descuentos","mcheques_emitidos_rechazados","ccheques_emitidos_rechazados","slope_TC_Total_Finiciomora_window_1","mcuenta_corriente_adicional","slope_TC_Total_madelantopesos_window_1","slope_TC_Total_madelantodolares_window_1","slope_TC_Total_cadelantosefectivo_window_1","slope_Visa_cadelantosefectivo_window_1","slope_Visa_madelantodolares_window_1","slope_Visa_madelantopesos_window_1","ctarjeta_master","cseguro_auto","cseguro_vida","cdescubierto_preacordado_delta_1_1","minversion1_dolares","mplazo_fijo_pesos","cprestamos_hipotecarios_delta_1_1","cprestamos_hipotecarios","slope_mcheques_depositados_rechazados_window_1"]

    # Top 40 de features de mayor importancia

    columnas_20_mas_importantes = ["ctrx_quarter","slope_ctrx_quarter_window","mpayroll","cpayroll_trx","mprestamos_personales","mcuentas_saldo","mpasivos_margen","mcaja_ahorro","ctrx_quarter_lag_1","slope_mprestamos_personales_window","slope_mcuentas_saldo_window","slope_mpayroll_window","slope_cpayroll_trx_window","mtarjeta_visa_consumo","mrentabilidad_annual","Visa_msaldopesos","ctarjeta_visa_transacciones","cliente_edad","mactivos_margen","ctarjeta_master","Master_fechaalta","Visa_fechaalta","Visa_Fvencimiento","Visa_msaldototal","TC_Total_mpagospesos","TC_Total_mpagominimo","mtransferencias_recibidas","mrentabilidad_annual_lag_1_1","TC_Total_fechaalta","Master_Fvencimiento","mactivos_margen_lag_1","numero_de_cliente","mcuenta_corriente_lag_1","mrentabilidad_annual_lag_1","mcaja_ahorro_lag_1_1","slope_mactivos_margen_window","mcuentas_saldo_delta_1","cliente_antiguedad","slope_mcuenta_corriente_window","mrentabilidad"]


    # 2. Feature Engineering
    df_fe = feature_engineering_tc_total(df)
    df_fe = generar_ctrx_features(df_fe)
    df_fe = feature_engineering_ratio(df_fe,columnas_20_mas_importantes)
    columnas_a_excluir = ["foto_mes","cliente_edad","numero_de_cliente","target"]
    atributos = [c for c in df.columns if c not in columnas_a_excluir]
    columnas_Master = [c for c in df.columns if c.startswith("Master_")]
    columnas_Visa = [c for c in df.columns if c.startswith("Visa_")]      
    columnas_categoricas = [c for c in df.columns if df[c].nunique() < 5]
    for i in (1,3):
        df_fe = feature_engineering_lag(df_fe, columnas=atributos, cant_lag=i)
    for i in (1,2):
        df_fe = feature_engineering_delta(df_fe, columnas=atributos, cant_delta=i)
    for i in (2,3):
        df_fe = feature_engineering_regr_slope_window(df_fe, columnas=atributos, ventana = i)

    # df_fe = feature_engineering_variables_canarios(df_fe)

    # Eliminar las columnas poco importantes de df_fe
    df_fe = df_fe.drop(columns=columnas_poco_importantes, errors='ignore')

    logger.info(f"Se eliminaron {len(columnas_poco_importantes)} columnas con importance_split <= 1")
    
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
    # df_fe_sampled = df_fe
    # df_fe_sampled = df_fe_sampled[~df_fe_sampled['foto_mes'].isin([202105,202106])]
    # logger.info(f"Excluyo de la muestra meses 5 y 6")
    # # df_fe_sampled.to_csv("data/competencia_fe_sampled.csv", index=False)

    # # 4. Ejecutar optimización (función simple)
    # study = optimizar_con_cv(df_fe_sampled, n_trials=80)
  
    # # 5. Análisis adicional
    # logger.info("=== ANÁLISIS DE RESULTADOS ===")
    # trials_df = study.trials_dataframe()
    # if len(trials_df) > 0:
    #     top_5 = trials_df.nlargest(5, 'value')
    #     logger.info("Top 5 mejores trials:")
    #     for idx, trial in top_5.iterrows():
    #         logger.info(
    #         f"Trial {int(trial['number'])}: "
    #         f"Ganancia = {trial['value']:,.0f} | "
    #         f"Parámetros: {trial['params']}")
  
    # logger.info("=== OPTIMIZACIÓN COMPLETADA ===")

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

    mejores_params = {'num_leaves': 46, 'learning_rate': 0.016377657023274192, 'min_data_in_leaf': 10, 'feature_fraction': 0.2503218637353462, 'bagging_fraction': 0.2352773905721117, 'num_boost_round': 1000}

    # mejores_params = cargar_mejores_hiperparametros()

  
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

