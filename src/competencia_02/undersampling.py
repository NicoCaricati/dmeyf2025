import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def undersample_clientes(df: pd.DataFrame, ratio: float, semilla: int = 555557) -> pd.DataFrame:
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
    logger.info(f"🔍 Iniciando undersampling con ratio={ratio} y semilla={semilla}")
    logger.info(f"➡️ DataFrame recibido con {df.shape[0]} filas y {df['numero_de_cliente'].nunique()} clientes únicos")

    if not (0 < ratio < 1):
        logger.warning("⚠️ Ratio inválido. Se devuelve el DataFrame original.")
        return df.copy()

    if 'target' not in df.columns or 'numero_de_cliente' not in df.columns:
        logger.error("❌ Faltan columnas necesarias: 'target' y/o 'numero_de_cliente'")
        return df.copy()

    # Filtrar valores válidos
    df = df[df['target'].isin([0, 1])].copy()
    logger.info(f"✅ Filtrado de target válido: {df.shape[0]} filas restantes")

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
    logger.info(f"📌 Clientes con target=1: {len(clientes_con_target1)}")

    # Clientes que siempre fueron 0
    clientes_siempre_0 = (
        df.loc[
            ~df["numero_de_cliente"].isin(clientes_con_target1),
            "numero_de_cliente",
        ]
        .unique()
    )
    logger.info(f"📌 Clientes siempre 0: {len(clientes_siempre_0)}")

    # Subsamplear clientes 0
    n_subsample = int(len(clientes_siempre_0) * ratio)
    if n_subsample == 0:
        logger.warning("⚠️ El número de clientes 0 a muestrear es 0. Ajustá el ratio.")
        return df.copy()

    clientes_siempre_0_sample = np.random.choice(
        clientes_siempre_0, n_subsample, replace=False
    )
    logger.info(f"🎯 Clientes 0 seleccionados: {len(clientes_siempre_0_sample)}")

    # Combinar ambos grupos
    clientes_final = np.concatenate(
        [clientes_con_target1.values, clientes_siempre_0_sample]
    )
    logger.info(f"📊 Total clientes seleccionados: {len(clientes_final)}")

    # Filtrar DataFrame
    df_filtrado = df[df["numero_de_cliente"].isin(clientes_final)].copy()
    logger.info(f"✅ DataFrame final tras undersampling: {df_filtrado.shape}")

    return df_filtrado
