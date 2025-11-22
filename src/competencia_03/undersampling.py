import numpy as np
import pandas as pd
import logging
from config import *

logger = logging.getLogger(__name__)

# def undersample_clientes(df: pd.DataFrame, ratio: float, semilla: int = 555557) -> pd.DataFrame:
#     """
#     Aplica undersampling a nivel cliente:
#     - Conserva todos los clientes que alguna vez tuvieron target=1 (en meses distintos de 202107 y 202108).
#     - Subsamplea clientes que siempre tuvieron target=0 (en meses distintos de 202107 y 202108).
#     - Mantiene todas las filas del DataFrame original para los clientes seleccionados.

#     Parámetros:
#     - df: DataFrame con columnas 'numero_de_cliente', 'target', 'foto_mes'.
#     - ratio: float entre 0 y 1 indicando proporción de clientes 0 a conservar.
#     - semilla: semilla para reproducibilidad.

#     Retorna:
#     - DataFrame filtrado con los clientes seleccionados.
#     """
#     logger.info(f"🔍 Iniciando undersampling con ratio={ratio} y semilla={semilla}")
#     logger.info(f"➡️ DataFrame recibido con {df.shape[0]} filas y {df['numero_de_cliente'].nunique()} clientes únicos")

#     if not (0 < ratio < 1):
#         logger.warning("⚠️ Ratio inválido. Se devuelve el DataFrame original.")
#         return df.copy()

#     if not {'target', 'numero_de_cliente', 'foto_mes'}.issubset(df.columns):
#         logger.error("❌ Faltan columnas necesarias: 'target', 'numero_de_cliente' y/o 'foto_mes'")
#         return df.copy()

#     # Filtrar solo meses válidos para el cálculo de target
#     df_filtrado_para_target = df[df['foto_mes'].isin(df['foto_mes'].unique()) & ~df['foto_mes'].isin([202107, 202108])]
#     df_filtrado_para_target = df_filtrado_para_target[df_filtrado_para_target['target'].isin([0, 1])].copy()

#     logger.info(f"✅ Filtrado para cálculo de target: {df_filtrado_para_target.shape[0]} filas")

#     np.random.seed(semilla)

#     # Clientes que alguna vez tuvieron target=1 (en meses válidos)
#     clientes_con_target1 = (
#         df_filtrado_para_target.groupby("numero_de_cliente")["target"]
#         .max()
#         .reset_index()
#     )
#     clientes_con_target1 = clientes_con_target1[
#         clientes_con_target1["target"] == 1
#     ]["numero_de_cliente"]
#     logger.info(f"📌 Clientes con target=1: {len(clientes_con_target1)}")

#     # Clientes que siempre fueron 0 (en meses válidos)
#     clientes_siempre_0 = (
#         df_filtrado_para_target.loc[
#             ~df_filtrado_para_target["numero_de_cliente"].isin(clientes_con_target1),
#             "numero_de_cliente",
#         ]
#         .unique()
#     )
#     logger.info(f"📌 Clientes siempre 0: {len(clientes_siempre_0)}")

#     # Subsamplear clientes 0
#     n_subsample = int(len(clientes_siempre_0) * ratio)
#     if n_subsample == 0:
#         logger.warning("⚠️ El número de clientes 0 a muestrear es 0. Ajustá el ratio.")
#         return df.copy()

#     clientes_siempre_0_sample = np.random.choice(
#         clientes_siempre_0, n_subsample, replace=False
#     )
#     logger.info(f"🎯 Clientes 0 seleccionados: {len(clientes_siempre_0_sample)}")

#     # Combinar ambos grupos
#     clientes_final = np.concatenate(
#         [clientes_con_target1.values, clientes_siempre_0_sample]
#     )
#     logger.info(f"📊 Total clientes seleccionados: {len(clientes_final)}")

#     # Filtrar DataFrame original (incluyendo todos los meses)
#     df_final = df[df["numero_de_cliente"].isin(clientes_final)].copy()
#     logger.info(f"✅ DataFrame final tras undersampling: {df_final.shape}")

#     return df_final



def undersample_clientes(df: pd.DataFrame, ratio: float, semilla: int = 555557) -> pd.DataFrame:
    logger.info(f"🔍 Iniciando undersampling con ratio={ratio} y semilla={semilla}")
    if not (0 < ratio < 1):
        logger.warning("⚠️ Ratio inválido. Se devuelve el DataFrame original.")
        return df.copy()

    df_filtrado = df[~df["foto_mes"].isin([202107, 202108]) & df["target"].isin([0, 1])].copy()

    clientes_con_target1 = df_filtrado.groupby("numero_de_cliente")["target"].max()
    clientes_con_target1 = clientes_con_target1[clientes_con_target1 == 1].index

    clientes_siempre_0 = df_filtrado.loc[~df_filtrado["numero_de_cliente"].isin(clientes_con_target1), "numero_de_cliente"].unique()
    np.random.seed(semilla)
    clientes_siempre_0_sample = np.random.choice(clientes_siempre_0, int(len(clientes_siempre_0) * ratio), replace=False)

    clientes_final = np.concatenate([clientes_con_target1, clientes_siempre_0_sample])
    df_final = df[df["numero_de_cliente"].isin(clientes_final)].copy()

    logger.info(f"✅ DataFrame final tras undersampling: {df_final.shape}")
    return df_final

