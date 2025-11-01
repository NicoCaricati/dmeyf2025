import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def undersample_clientes(df: pd.DataFrame, ratio: float, semilla: int = SEMILLA[0]) -> pd.DataFrame:
    """
    Aplica undersampling a nivel cliente:
    - Conserva todos los clientes que alguna vez tuvieron target=1.
    - Subsamplea clientes que siempre tuvieron target=0 según el ratio indicado.

    Parámetros:
    - df: DataFrame con columnas 'numero_de_cliente' y 'target'.
    - ratio: float entre 0 y 1 indicando proporción de clientes 0 a conservar.
    - semilla: semilla para reproducibilidad.

    Retorna:
    - DataFrame filtrado con los clientes seleccionados.
    """
    if not (0 < ratio < 1):
        logger.warning("Ratio inválido. Se devuelve el DataFrame original.")
        return df.copy()

    np.random.seed(semilla)

    # Clientes que alguna vez tuvieron target=1
    clientes_con_target1 = (
        df.groupby("numero_de_cliente")["target"]
        .max()
        .reset_index()
    )
    clientes_con_target1 = clientes_con_target1[
        clientes_con_target1["target"] == 1
    ]["numero_de_cliente"]

    # Clientes que siempre fueron 0
    clientes_siempre_0 = (
        df.loc[
            ~df["numero_de_cliente"].isin(clientes_con_target1),
            "numero_de_cliente",
        ]
        .unique()
    )

    # Subsamplear clientes 0
    n_subsample = int(len(clientes_siempre_0) * ratio)
    clientes_siempre_0_sample = np.random.choice(
        clientes_siempre_0, n_subsample, replace=False
    )

    # Combinar ambos grupos
    clientes_final = np.concatenate(
        [clientes_con_target1.values, clientes_siempre_0_sample]
    )

    # Filtrar DataFrame
    df_filtrado = df[df["numero_de_cliente"].isin(clientes_final)].copy()

    logger.debug(
        f"Undersampling aplicado: {len(clientes_con_target1)} clientes con target=1 "
        f"+ {len(clientes_siempre_0_sample)} clientes 0 (de {len(clientes_siempre_0)} posibles) "
        f"→ total {len(clientes_final)} clientes en train."
    )

    return df_filtrado
