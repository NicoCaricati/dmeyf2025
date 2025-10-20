import numpy as np
import pandas as pd
from config import *
import logging
import polars as pl

logger = logging.getLogger(__name__)

def calcular_ganancia(y_true, y_pred):
    """
    Calcula la ganancia total usando la función de ganancia de la competencia.
 
    Args:
        y_true: Valores reales (0 o 1)
        y_pred: Predicciones (0 o 1)
  
    Returns:
        float: Ganancia total
    """
    # Convertir a numpy arrays si es necesario
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
  
    # Calcular ganancia vectorizada usando configuración
    # Verdaderos positivos: y_true=1 y y_pred=1 -> ganancia
    # Falsos positivos: y_true=0 y y_pred=1 -> costo
    # Verdaderos negativos y falsos negativos: ganancia = 0
  
    ganancia_total = np.sum(
        ((y_true == 1) & (y_pred == 1)) * GANANCIA_ACIERTO +  # TP
        ((y_true == 0) & (y_pred == 1)) * (-COSTO_ESTIMULO)   # FP
    )
  
    logger.debug(f"Ganancia calculada: {ganancia_total:,.0f} "
                f"(GANANCIA_ACIERTO={GANANCIA_ACIERTO}, COSTO_ESTIMULO={COSTO_ESTIMULO})")
  
    return ganancia_total

def ganancia_lgb_binary(y_pred, y_true):
    """
    Función de ganancia para LightGBM en clasificación binaria.
    Compatible con callbacks de LightGBM.
  
    Args:
        y_pred: Predicciones de probabilidad del modelo
        y_true: Dataset de LightGBM con labels verdaderos
  
    Returns:
        tuple: (eval_name, eval_result, is_higher_better)
    """
    # Obtener labels verdaderos
    y_true_labels = y_true.get_label()
  
    # Convertir probabilidades a predicciones binarias (umbral 0.5)
    y_pred_binary = (y_pred > UMBRAL).astype(int)
  
    # Calcular ganancia usando configuración
    ganancia_total = calcular_ganancia(y_true_labels, y_pred_binary)
  
    # Retornar en formato esperado por LightGBM
    return 'ganancia', ganancia_total, True  # True = higher is better


def ganancia_evaluator(y_pred, y_true) -> float:
    """
    Calcula la ganancia máxima acumulada de un modelo binario en base a
    probabilidades predichas y valores verdaderos.
    Compatible tanto con uso interno en LightGBM (feval) como externo (post-test).

    Parameters
    ----------
    y_pred : array-like
        Probabilidades predichas por el modelo (LightGBM pasa este vector).
    y_true : array-like o lightgbm.Dataset
        Etiquetas verdaderas. Si es un Dataset de LightGBM, se accede con .get_label().

    Returns
    -------
    tuple o float
        - En contexto LightGBM: ('ganancia', valor, True)
        - En contexto externo: valor numérico de la ganancia máxima
    """

    # Permitir uso interno y externo
    is_lightgbm = hasattr(y_true, "get_label")
    if is_lightgbm:
        y_true = y_true.get_label()

    # Convertir a dataframe de Polars
    df_eval = pl.DataFrame({
        "y_true": y_true,
        "y_pred_proba": y_pred
    })

    # Ordenar por probabilidad descendente
    df_ordenado = df_eval.sort("y_pred_proba", descending=True)

    # Calcular ganancia individual
    df_ordenado = df_ordenado.with_columns([
        pl.when(pl.col("y_true") == 1)
        .then(GANANCIA_ACIERTO)
        .otherwise(-COSTO_ESTIMULO)
        .alias("ganancia_individual")
    ])

    # Ganancia acumulada
    df_ordenado = df_ordenado.with_columns([
        pl.col("ganancia_individual").cum_sum().alias("ganancia_acumulada")
    ])

    # Ganancia máxima alcanzada
    ganancia_maxima = df_ordenado["ganancia_acumulada"].max()

    # Si se usa en LightGBM, devolver en formato feval
    if is_lightgbm:
        return ('ganancia', ganancia_maxima, True)
    else:
        return ganancia_maxima

