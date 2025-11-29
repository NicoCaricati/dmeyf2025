import numpy as np
import pandas as pd
from config import *
import logging
import polars as pl

logger = logging.getLogger(__name__)

def calcular_ganancia(y_true, y_pred):
    """
    Calcula la ganancia total usando la función de ganancia de la competencia.
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    assert len(y_true) == len(y_pred), "Error: y_true y y_pred tienen distinta longitud"

    ganancia_total = np.sum(
        ((y_true == 1) & (y_pred == 1)) * GANANCIA_ACIERTO +
        ((y_true == 0) & (y_pred == 1)) * (-COSTO_ESTIMULO)
    )

    logger.debug(f"Ganancia calculada: {ganancia_total:,.0f} (TP={((y_true == 1) & (y_pred == 1)).sum()}, FP={((y_true == 0) & (y_pred == 1)).sum()})")
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


def calcular_ganancia_top_k(y_true, y_pred_proba, k=10000):
    """
    Calcula la ganancia total considerando como positivos los k casos con mayor probabilidad.
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred_proba, pd.Series):
        y_pred_proba = y_pred_proba.values

    assert len(y_true) == len(y_pred_proba), "Error: y_true y y_pred_proba tienen distinta longitud"

    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred_proba": y_pred_proba
    }).sort_values("y_pred_proba", ascending=False).reset_index(drop=True)

    df["predict"] = 0
    df.loc[:k - 1, "predict"] = 1

    tp = ((df["y_true"] == 1) & (df["predict"] == 1)).sum()
    fp = ((df["y_true"] == 0) & (df["predict"] == 1)).sum()

    ganancia_total = tp * GANANCIA_ACIERTO - fp * COSTO_ESTIMULO

    logger.debug(f"Ganancia top_k={k}: TP={tp}, FP={fp}, Total={ganancia_total:,.0f}")
    return ganancia_total