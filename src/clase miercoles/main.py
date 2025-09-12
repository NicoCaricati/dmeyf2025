import pandas as pd
import os
import datetime
import logging
import numpy as np
import optuna


from loader import cargar_datos
from features import feature_engineering_lag
from features import feature_engineering_delta
from features import feature_engineering_tasa_variacion
from features import feature_engineering_tc_total
from features import feature_engineering_regr_slope_window
from features import feature_engineering_ratio

## config basico logging
os.makedirs("logs", exist_ok=True)

fecha = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
monbre_log = f"log_{fecha}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/{monbre_log}", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

## Funcion principal
def main():
    logger.info("Inicio de ejecucion.")

    #00 Cargar datos
    os.makedirs("data", exist_ok=True)
    path = "data/competencia_03.csv"
    df = cargar_datos(path)   

    #01 Feature Engineering
    df = feature_engineering_tc_total(df)
    columnas_a_excluir = ["foto_mes", "cliente_antiguedad","cliente_edad","numero_de_cliente","target"]
    atributos = [c for c in df.columns if c not in columnas_a_excluir]
    columnas_Master = [c for c in df.columns if c.startswith("Master_")]
    columnas_Visa = [c for c in df.columns if c.startswith("Visa_")]      
    columnas_categoricas = [c for c in df.columns if df[c].nunique() < 5]
    columnas_a_excluir_ratios = ["foto_mes", "cliente_antiguedad","cliente_edad","numero_de_cliente","target",columnas_Master, columnas_Visa, columnas_categoricas] 
    # atributos_ratios = [c for c in df.columns if c not in columnas_a_excluir_ratios]
    atributos_ratios = ['mactivos_margen' , 'matm' , 'matm_other' , 'mautoservicio' , 'mcaja_ahorro' , 'mcaja_ahorro_adicional' , 'mcaja_ahorro_dolares' , 'mcajeros_propios_descuentos' , 'mcheques_depositados' , 'mcheques_depositados_rechazados' , 'mcheques_emitidos' , 'mcheques_emitidos_rechazados' , 'mcomisiones' , 'mcomisiones_mantenimiento' , 'mcomisiones_otras' , 'mcuenta_corriente' , 'mcuenta_corriente_adicional' , 'mcuenta_debitos_automaticos' , 'mcuentas_saldo' , 'mextraccion_autoservicio' , 'mforex_buy' , 'mforex_sell' , 'minversion1_dolares' , 'minversion1_pesos' , 'minversion2' , 'mpagodeservicios' , 'mpagomiscuentas' , 'mpasivos_margen' , 'mpayroll' , 'mpayroll2' , 'mplazo_fijo_dolares' , 'mplazo_fijo_pesos' , 'mprestamos_hipotecarios' , 'mprestamos_personales' , 'mprestamos_prendarios' , 'mrentabilidad' , 'mrentabilidad_annual' , 'mtarjeta_master_consumo' , 'mtarjeta_master_descuentos' , 'mtarjeta_visa_consumo' , 'mtarjeta_visa_descuentos' , 'mtransferencias_emitidas' , 'mtransferencias_recibidas' , 'mttarjeta_master_debitos_automaticos' , 'mttarjeta_visa_debitos_automaticos']
    df = feature_engineering_lag(df, columnas=atributos, cant_lag= 1)
    df = feature_engineering_delta(df, columnas=atributos, cant_delta= 1)
    df = feature_engineering_tasa_variacion(df, columnas=atributos, cant_tasa_variacion= 1)
    df = feature_engineering_regr_slope_window(df, columnas=atributos, ventana = 3)
    # df = feature_engineering_ratio(df, columnas=atributos_ratios)
    
    #02 Guardar datos
    path = "data/competencia_fe.csv"
    df.to_csv(path, index=False)
  
    logger.info(f">>> Ejecución finalizada. Revisar logs para mas detalles.{monbre_log}")

if __name__ == "__main__":
    main()
    


