# main.py

import logging
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
from config import STUDY_NAME, BUCKET_NAME
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def guardar_predicciones_finales(resultados, nombre_archivo=None):
    """
    Guarda DataFrames de predicción en CSV.
    Puede recibir un dict de DataFrames o un DataFrame único.
    """
    os.makedirs("predict", exist_ok=True)

    if isinstance(resultados, pd.DataFrame):
        resultados_dict = {"top_k": resultados}
    else:
        resultados_dict = resultados

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rutas = {}

    for tipo, df in resultados_dict.items():
        ruta = f"predict/{nombre_archivo}_{tipo}_{timestamp}.csv"
        df.to_csv(ruta, index=False)
        df.to_csv(f"{BUCKET_NAME}/{STUDY_NAME}/{nombre_archivo}_{tipo}_{timestamp}.csv")  
        rutas[tipo] = ruta

        logger.info(f"Predicciones ({tipo}) guardadas en: {ruta}")
        logger.info(f"  Columnas: {list(df.columns)}")
        logger.info(f"  Registros: {len(df):,}")
        logger.info(f"  Primeras filas:\n{df.head()}")

    return rutas

