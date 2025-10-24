import pandas as pd
import os
import logging
from datetime import datetime
from config import STUDY_NAME

logger = logging.getLogger(__name__)

def guardar_predicciones_finales(resultados_dict: dict, nombre_archivo=None) -> dict:
    """
    Guarda los distintos DataFrames de predicción en CSV (umbral, top_k si aplica).

    Args:
        resultados_dict: Diccionario con DataFrames ('umbral', 'top_k')
        nombre_archivo: Nombre base del archivo (si es None, usa STUDY_NAME)

    Returns:
        dict: {'umbral': ruta_csv, 'top_k': ruta_csv (si aplica)}
    """
    os.makedirs("predict", exist_ok=True)
    os.makedirs(f"../../../buckets/b1/Compe_02/{study_name}", exist_ok=True)

    if nombre_archivo is None:
        nombre_archivo = STUDY_NAME

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rutas = {}

    study_name = STUDY_NAME

    for tipo, df in resultados_dict.items():
        ruta = f"predict/{nombre_archivo}_{tipo}_{timestamp}.csv"
        ruta_2 = f"../../../buckets/b1/Compe_02/{study_name}/{nombre_archivo}_{tipo}_{timestamp}.csv"
        df.to_csv(ruta, index=False)
        df.to_csv(ruta_2, index=False)
        rutas[tipo] = ruta

        logger.info(f"Predicciones ({tipo}) guardadas en: {ruta}")
        logger.info(f"  Columnas: {list(df.columns)}")
        logger.info(f"  Registros: {len(df):,}")
        logger.info(f"  Primeras filas:\n{df.head()}")

    return rutas