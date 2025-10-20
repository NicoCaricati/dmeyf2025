import pandas as pd
import logging
import duckdb

from config import MES_TRAIN, MES_VALIDACION, MES_TEST, FINAL_PREDIC

logger = logging.getLogger("__name__")

""" ## Funcion para cargar datos
def cargar_datos(path: str) -> pd.DataFrame | None:

    '''
    Carga el CSV desde CSV y retorna un pandas.DataFrame.
    '''


    logger.info(f"Cargando dataset desde {path}")
    try:
        df = pd.read_csv(path)
        # Filtrar solo meses de interés (train, validación, test)
        df = df[df['foto_mes'].isin(MES_TRAIN + [MES_VALIDACION] + [MES_TEST] + [FINAL_PREDIC])]
        logger.info(f"Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas")
        return df
    except Exception as e:
        logger.error(f"Error al cargar el dataset: {e}")
        raise

 """

""" import zipfile
import os
from kaggle.api.kaggle_api_extended import KaggleApi

logger = logging.getLogger("__name__")

def cargar_datos_kaggle(competition: str, filename: str, extract_path: str = "data") -> pd.DataFrame:
    Descarga un dataset de una competencia de Kaggle y lo carga en un DataFrame.

    Args:
        competition (str): nombre de la competencia (ej: "dm-ey-f-2025-primera")
        filename (str): nombre del archivo CSV dentro del zip
        extract_path (str): carpeta destino para extraer los archivos

    Returns:
        pd.DataFrame: dataset cargado
    try:
        # Inicializar API de Kaggle
        api = KaggleApi()
        api.authenticate()

        # Crear carpeta destino si no existe
        os.makedirs(extract_path, exist_ok=True)

        zip_path = os.path.join(extract_path, f"{competition}.zip")

        # Descargar si aún no existe
        if not os.path.exists(zip_path):
            logger.info(f"Descargando dataset de {competition}...")
            api.competition_download_files(competition, path=extract_path)
            logger.info("Descarga completada")

        # Extraer
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_path)

        # Cargar CSV
        csv_path = os.path.join(extract_path, filename)
        df = pd.read_csv(csv_path)
        logger.info(f"Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas")

        return df

    except Exception as e:
        logger.error(f"Error al cargar dataset desde Kaggle: {e}")
        raise

 """

logger = logging.getLogger(__name__)

def cargar_datos(path: str) -> pd.DataFrame | None:
    '''
    Carga el CSV crudo (sin target), calcula la columna target con SQL,
    hace el join para conservar todas las columnas originales
    y retorna un pandas.DataFrame filtrado por meses de interés.
    '''

    logger.info(f"Cargando dataset desde {path}")
    try:
        con = duckdb.connect(database=':memory:')

        # Registrar CSV como tabla cruda
        con.execute(f"""
            CREATE OR REPLACE TABLE competencia_01_crudo AS 
            SELECT * FROM read_csv_auto('{path}')
        """)

        # Crear tabla con target
        con.execute("""
            CREATE OR REPLACE TABLE competencia_01 AS 
            SELECT numero_de_cliente,
                   foto_mes,
                   CASE 
                       WHEN foto_mes_1 IS NULL THEN 'BAJA+1'
                       WHEN foto_mes_2 IS NULL THEN 'BAJA+2'
                       ELSE 'CONTINUA'
                   END AS target
            FROM (
                SELECT numero_de_cliente,
                       foto_mes,
                       LEAD(foto_mes, 1, NULL) OVER (
                           PARTITION BY numero_de_cliente ORDER BY foto_mes
                       ) AS foto_mes_1,
                       LEAD(foto_mes, 2, NULL) OVER (
                           PARTITION BY numero_de_cliente ORDER BY foto_mes
                       ) AS foto_mes_2
                FROM competencia_01_crudo
            ) a
        """)

        # Hacer join para quedarnos con todas las columnas originales + target
        con.execute("""
            CREATE OR REPLACE TABLE competencia_03 AS (
                SELECT a.*, b.target
                FROM competencia_01_crudo a
                LEFT JOIN competencia_01 b
                USING (numero_de_cliente, foto_mes)
            )
        """)

        # Pasar a pandas
        df = con.execute("SELECT * FROM competencia_03").fetchdf()

        # # Filtrar meses de interés
        # df = df[df['foto_mes'].isin(MES_TRAIN + [MES_VALIDACION] + [MES_TEST] + [FINAL_PREDIC])]

        logger.info(f"Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas")
        return df

    except Exception as e:
        logger.error(f"Error al cargar el dataset: {e}")
        raise

def convertir_clase_ternaria_a_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte clase_ternaria a target binario reemplazando en el mismo atributo:
    - CONTINUA = 0
    - BAJA+1 y BAJA+2 = 1
  
    Args:
        df: DataFrame con columna 'clase_ternaria'
  
    Returns:
        pd.DataFrame: DataFrame con clase_ternaria convertida a valores binarios (0, 1)
    """
    # Crear copia del DataFrame para no modificar el original
    df_result = df.copy()

    # Seteo para meses 5 y 6 target como null
    if 'foto_mes' in df_result.columns and 'target' in df_result.columns:
        df_result.loc[df_result['foto_mes'].isin([202105, 202106]), 'target'] = None
  
    # Contar valores originales para logging
    n_continua_orig = (df_result['target'] == 'CONTINUA').sum()
    n_baja1_orig = (df_result['target'] == 'BAJA+1').sum()
    n_baja2_orig = (df_result['target'] == 'BAJA+2').sum()
  
    # Convertir clase_ternaria a binario en el mismo atributo
    df_result['target_to_calculate_gan'] = df_result['target'].map({
        'CONTINUA': 0,
        'BAJA+1': 0,
        'BAJA+2': 1
    })
  

    df_result['target'] = df_result['target'].map({
        'CONTINUA': 0,
        'BAJA+1': 1,
        'BAJA+2': 1
    })


  
    # Log de la conversión
    n_ceros = (df_result['target'] == 0).sum()
    n_unos = (df_result['target'] == 1).sum()
  
    logger.info(f"Conversión completada:")
    logger.info(f"  Mapeo utilizado: CONTINUA -> 0, BAJA+1 -> 1, BAJA+2 -> 1")
    logger.info(f"  Original - CONTINUA: {n_continua_orig}, BAJA+1: {n_baja1_orig}, BAJA+2: {n_baja2_orig}")
    logger.info(f"  Binario - 0: {n_ceros}, 1: {n_unos}")
    logger.info(f"  Distribución: {n_unos/(n_ceros + n_unos)*100:.2f}% casos positivos")
  
    return df_result

