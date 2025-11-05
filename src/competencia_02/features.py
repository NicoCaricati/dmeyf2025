# features

import pandas as pd
import duckdb
import logging
import numpy as np
import polars as pl
from itertools import combinations
from config import GRUPOS_VARIABLES
import re

logger = logging.getLogger("__name__")

def feature_engineering_lag(df: pd.DataFrame, columnas: list[str], cant_lag: int = 1) -> pd.DataFrame:
    """
    Genera variables de lag para los atributos especificados utilizando SQL.
  
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list
        Lista de atributos para los cuales generar lags. Si es None, no se generan lags.
    cant_lag : int, default=1
        Cantidad de lags a generar para cada atributo
  
    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de lag agregadas
    """

    logger.info(f"Realizando feature engineering con {cant_lag} lags para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar lags")
        return df
  
    # Construir la consulta SQL
    columnas_sql = ", ".join(df.columns)
    sql = f"SELECT {columnas_sql}"

    # Agregar los lags para los atributos especificados
# Agregar solo el lag especificado para cada atributo
    for attr in columnas:
        if attr in df.columns:
            sql += f", lag({attr}, {cant_lag}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_lag_{cant_lag}"
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    # Completar la consulta
    sql += " FROM df"

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df()
    con.close()

    print(df.head())
  
    logger.info(f"Feature engineering completado. DataFrame resultante con {df.shape[1]} columnas")

    return df



def feature_engineering_delta(df: pd.DataFrame, columnas: list[str], cant_delta: int = 1) -> pd.DataFrame:
    """
    Genera variables de delta para los atributos especificados utilizando SQL.
  
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list
        Lista de atributos para los cuales generar deltas. Si es None, no se generan deltas.
    cant_delta : int, default=1
        Cantidad de deltas a generar para cada atributo
  
    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de delta agregadas
    """

    logger.info(f"Realizando feature engineering con {cant_delta} deltas para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar deltas")
        return df
  
    # Construir la consulta SQL
    columnas_sql = ", ".join(df.columns)
    sql = f"SELECT {columnas_sql}"
  
    # Agregar solo el delta especificado para cada atributo
    for attr in columnas:
        if attr in df.columns:
            sql += f", {attr} - LAG({attr}, {cant_delta}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_delta_{cant_delta}"
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    # Completar la consulta
    sql += " FROM df"

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df()
    con.close()

    print(df.head())
  
    logger.info(f"Feature engineering completado. DataFrame resultante con {df.shape[1]} columnas")

    return df


def feature_engineering_tc_total(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera variables agregadas TC_Total combinando columnas de Master y Visa
    según el método de agregación definido (Sum, Max, Min).

    - Sum: se suman los valores de Master y Visa
    - Max: se toma el máximo entre Master y Visa
    - Min: se toma el mínimo entre Master y Visa

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada con columnas Master_* y Visa_*

    Returns
    -------
    pd.DataFrame
        DataFrame con nuevas variables TC_Total_*
    """



    # --- SUM ---
    sum_cols = [
        "cadelantosefectivo", "cconsumos", "madelantodolares", "madelantopesos",
        "mconsumosdolares", "mconsumospesos", "mconsumototal",
        "mfinanciacion_limite", "mlimitecompra", "mpagado", "mpagominimo",
        "mpagosdolares", "mpagospesos", "msaldodolares", "msaldopesos", "msaldototal"
    ]
    for col in sum_cols:
        master_col, visa_col = f"Master_{col}", f"Visa_{col}"
        if master_col in df.columns and visa_col in df.columns:
            df[f"TC_Total_{col}"] = df[master_col].fillna(0) + df[visa_col].fillna(0)

    # --- MAX ---
    max_cols = ["delinquency", "status"]
    for col in max_cols:
        master_col, visa_col = f"Master_{col}", f"Visa_{col}"
        if master_col in df.columns and visa_col in df.columns:
            df[f"TC_Total_{col}"] = np.maximum(df[master_col], df[visa_col])

    # --- MIN ---
    min_cols = ["fechaalta", "Finiciomora", "fultimo_cierre", "Fvencimiento"]
    for col in min_cols:
        master_col, visa_col = f"Master_{col}", f"Visa_{col}"
        if master_col in df.columns and visa_col in df.columns:
            df[f"TC_Total_{col}"] = np.minimum(df[master_col], df[visa_col])

    logger.info(f"Variables TC_Total generadas. DataFrame resultante con {df.shape[1]} columnas")

    return df
    
def feature_engineering_regr_slope_window(df: pd.DataFrame, columnas: list[str], ventana: int = 3) -> pd.DataFrame:
        """
        Calcula la pendiente de regresión (slope) de cada atributo especificado respecto al tiempo (foto_mes)
        para cada cliente, usando una ventana móvil de meses.
    
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con los datos, debe contener 'numero_de_cliente' y 'foto_mes'
        columnas : list[str]
            Lista de columnas sobre las cuales calcular la pendiente
        ventana : int, default=12
            Cantidad de meses de la ventana móvil
    
        Returns
        -------
        pd.DataFrame
            DataFrame con nuevas columnas 'slope_<atributo>_window' agregadas
        """
    
        logger = logging.getLogger(__name__)
        logger.info(f"Calculando slope con ventana móvil de {ventana} meses para {len(columnas)} atributos")
    
        if not columnas:
            logger.warning("No se especificaron atributos para calcular slope")
            return df
    
        con = duckdb.connect(database=":memory:")
        con.register("df", df)
    
        # Construir expresiones de slope para cada columna
        slope_exprs = []
        for col in columnas:
            if col in df.columns:
                # slope = cov(x, y) / var(x) usando ventana de ROWS
                expr = f"""
                    AVG(foto_mes * {col}) OVER (
                        PARTITION BY numero_de_cliente
                        ORDER BY foto_mes
                        ROWS BETWEEN {ventana-1} PRECEDING AND CURRENT ROW
                    )
                    -
                    AVG(foto_mes) OVER (
                        PARTITION BY numero_de_cliente
                        ORDER BY foto_mes
                        ROWS BETWEEN {ventana-1} PRECEDING AND CURRENT ROW
                    )
                    *
                    AVG({col}) OVER (
                        PARTITION BY numero_de_cliente
                        ORDER BY foto_mes
                        ROWS BETWEEN {ventana-1} PRECEDING AND CURRENT ROW
                    )
                    /
                    (
                        AVG(foto_mes * foto_mes) OVER (
                            PARTITION BY numero_de_cliente
                            ORDER BY foto_mes
                            ROWS BETWEEN {ventana-1} PRECEDING AND CURRENT ROW
                        )
                        -
                        AVG(foto_mes) OVER (
                            PARTITION BY numero_de_cliente
                            ORDER BY foto_mes
                            ROWS BETWEEN {ventana-1} PRECEDING AND CURRENT ROW
                        ) * 
                        AVG(foto_mes) OVER (
                            PARTITION BY numero_de_cliente
                            ORDER BY foto_mes
                            ROWS BETWEEN {ventana-1} PRECEDING AND CURRENT ROW
                        )
                    ) AS slope_{col}_{ventana}_window
                """
                slope_exprs.append(expr)
            else:
                logger.warning(f"Columna {col} no encontrada en df")
    
        sql = f"""
            SELECT *,
                   {', '.join(slope_exprs)}
            FROM df
        """
    
        logger.debug(f"Consulta SQL slope ventana: {sql[:500]}...")  # Limitar tamaño del log
        df = con.execute(sql).df()
        con.close()
    
        logger.info(f"Slope con ventana móvil calculado. DataFrame resultante con {df.shape[1]} columnas")
        return df

def feature_engineering_ratio(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """
    Genera nuevas variables dividiendo todas las columnas especificadas entre sí (ratios),
    evitando divisiones por cero y duplicados, y las agrega al DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list[str]
        Lista de columnas para generar todos los ratios posibles entre sí.
        Ej: ['mcuentas_saldo', 'mpayroll', 'mconsumototal']

    Returns
    -------
    pd.DataFrame
        DataFrame con nuevas columnas 'ratio_<numerador>_<denominador>'
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Generando ratios para {len(columnas)} columnas")

    # Generar todas las combinaciones sin repetición
    for numerador, denominador in combinations(columnas, 2):
        if numerador in df.columns and denominador in df.columns:
            col_name = f"ratio_{numerador}_{denominador}"
            df[col_name] = df[numerador] / df[denominador].replace(0, np.nan)

            # col_name_inv = f"ratio_{denominador}_{numerador}"
            # df[col_name_inv] = df[denominador] / df[numerador].replace(0, np.nan)
        
    logger.info(f"Feature engineering de ratios completado. DataFrame ahora tiene {df.shape[1]} columnas y {df.shape[0]} filas")
    return df

# variables canarios como variables uniformes (0,1) q voy a usar para cortar variables de importancia menor a lo random
def feature_engineering_variables_canarios(df: pd.DataFrame, n_canarios: int = 50) -> pd.DataFrame:
    """
    Agrega variables canarias al DataFrame. Las variables canarias son columnas con valores aleatorios
    que sirven como referencia para evaluar la importancia de otras variables en modelos predictivos.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame al cual se le agregarán las variables canarias.
    n_canarios : int, default=5
        Cantidad de variables canarias a generar.

    Returns
    -------
    pd.DataFrame
        DataFrame con las variables canarias agregadas.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Agregando {n_canarios} variables canarias al DataFrame")

    for i in range(n_canarios):
        col_name = f"canario_{i+1}"
        df[col_name] = np.random.uniform(0, 1, size=df.shape[0])
        logger.debug(f"Variable canaria creada: {col_name}")

    logger.info(f"Variables canarias agregadas. DataFrame ahora tiene {df.shape[1]} columnas y {df.shape[0]} filas")
    return df


def generar_ctrx_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera ctrx_30d, ctrx_60d, ctrx_90d, ctrx_120d usando SQL en DuckDB.
    """
    sql = """
    SELECT *,
        -- ctrx_30d = suma mensual de transacciones
        coalesce(ctarjeta_debito_transacciones,0)
        + coalesce(ctarjeta_visa_transacciones,0)
        + coalesce(ctarjeta_master_transacciones,0)
        + coalesce(ccajas_transacciones,0) AS ctrx_30d,

        -- rolling sums
        sum(
            coalesce(ctarjeta_debito_transacciones,0)
            + coalesce(ctarjeta_visa_transacciones,0)
            + coalesce(ctarjeta_master_transacciones,0)
            + coalesce(ccajas_transacciones,0)
        ) OVER (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes
            ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
        ) AS ctrx_60d,

        sum(
            coalesce(ctarjeta_debito_transacciones,0)
            + coalesce(ctarjeta_visa_transacciones,0)
            + coalesce(ctarjeta_master_transacciones,0)
            + coalesce(ccajas_transacciones,0)
        ) OVER (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS ctrx_90d,

    FROM df
    """

    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df_out = con.execute(sql).df()
    con.close()
    return df_out

def calculate_psi(expected, actual, buckets=10):
    expected = np.array(expected)
    actual = np.array(actual)
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))

    def get_counts(data):
        counts = np.histogram(data, bins=breakpoints)[0]
        return counts / len(data)

    expected_percents = get_counts(expected)
    actual_percents = get_counts(actual)

    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)

    psi_values = (expected_percents - actual_percents) * np.log(expected_percents / actual_percents)
    return np.sum(psi_values)


def psi_by_columns(df, cols, fecha_base=202104, fecha_prod=202106, fotomes_col="fotomes"):
    results = {}
    for col in cols:
        base = df.loc[df[fotomes_col] == fecha_base, col].dropna()
        prod = df.loc[df[fotomes_col] == fecha_prod, col].dropna()
        if len(base) > 0 and len(prod) > 0:
            results[col] = calculate_psi(base, prod)
        else:
            results[col] = None
    return pd.Series(results, name="PSI")



def feature_engineering_mpayroll_corregida(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige el monto de payroll en meses de aguinaldo.
    Para fotomes 202106, si mpayroll > 0, multiplica por 0.67.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    mask = (df["foto_mes"] == 202106) & (df["mpayroll"] > 0)
    df["mpayroll_corregida"] = df["mpayroll"]
    df.loc[mask, "mpayroll_corregida"] = df.loc[mask, "mpayroll"] * 0.67
    return df


def feature_engineering_cpayroll_trx_corregida(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige el conteo de transacciones de payroll en meses de aguinaldo.
    Para foto_mes  202106, si mpayroll > 0, resta una transacción.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    mask = (df["foto_mes"] == 202106) & (df["mpayroll"] > 0)
    df["cpayroll_trx_corregida"] = df["cpayroll_trx"]
    df.loc[mask, "cpayroll_trx_corregida"] = df.loc[mask, "cpayroll_trx"] - 1
    return df




def variables_aux(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ===============================
    # SANEAR COLUMNAS NUMÉRICAS
    # ===============================
    columnas_numericas = [
        # Actividad
        "ctarjeta_visa_transacciones", "ctarjeta_master_transacciones",
        "ctarjeta_debito_transacciones", "chomebanking_transacciones", "cmobile_app_trx",
        "ctransferencias_emitidas", "ctransferencias_recibidas",
        # Rentabilidad
        "mrentabilidad_annual", "cliente_antiguedad", "mactivos_margen",
        "mpasivos_margen", "mrentabilidad", "mcomisiones", "cproductos",
        # Crédito
        "Visa_msaldototal", "Master_msaldototal", "Visa_mlimitecompra",
        "Master_mlimitecompra", "Visa_delinquency", "Master_delinquency",
        "Visa_Finiciomora", "Master_Finiciomora",
        # Ahorro / inversión
        "mcaja_ahorro", "mcuenta_corriente", "mplazo_fijo_pesos", "minversion1_pesos",
        "minversion2", "mcaja_ahorro_dolares", "mplazo_fijo_dolares", "minversion1_dolares",
        # Movimientos
        "mtransferencias_recibidas", "mtransferencias_emitidas",
        "mcheques_depositados", "mcheques_emitidos", "mcuentas_saldo",
        "mextraccion_autoservicio",
        # Ingresos
        "mpayroll", "mpayroll2", "cpayroll_trx",
        # Productos / canales
        "thomebanking", "tcallcenter", "tmobile_app", "tcuentas",
        "ctarjeta_visa", "ctarjeta_master",
        # Otros
        "cliente_vip"
    ]

    # Asegurar que solo se trabajen columnas existentes
    cols_existentes = [c for c in columnas_numericas if c in df.columns]

    for c in cols_existentes:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df[cols_existentes] = df[cols_existentes].fillna(0)

    # ===============================
    # 1. ACTIVIDAD / ENGAGEMENT
    # ===============================
    df["actividad_total"] = (
        df["ctarjeta_visa_transacciones"] +
        df["ctarjeta_master_transacciones"] +
        df["ctarjeta_debito_transacciones"] +
        df["chomebanking_transacciones"] +
        df["cmobile_app_trx"] +
        df["ctransferencias_emitidas"] +
        df["ctransferencias_recibidas"]
    )

    df["uso_digital_ratio"] = (
        (df["chomebanking_transacciones"] + df["cmobile_app_trx"]) /
        (1 + df["actividad_total"])
    )

    df["uso_tarjeta_ratio"] = (
        (df["ctarjeta_visa_transacciones"] + df["ctarjeta_master_transacciones"]) /
        (1 + df["actividad_total"])
    )

    df["canales_activos"] = (
        df["thomebanking"] + df["tcallcenter"] + df["tmobile_app"]
    )

    df["usa_app_o_homebanking"] = (
        ((df["tmobile_app"] == 1) | (df["thomebanking"] == 1)).astype(int)
    )

    # ===============================
    # 2. RENTABILIDAD / VALOR
    # ===============================
    df["rentabilidad_media_mensual"] = df["mrentabilidad_annual"] / (1 + df["cliente_antiguedad"])
    df["margen_total"] = df["mactivos_margen"] + df["mpasivos_margen"]
    df["rentabilidad_vs_margen"] = df["mrentabilidad"] / (1 + df["margen_total"])
    df["margen_por_producto"] = df["margen_total"] / (1 + df["cproductos"])
    df["comisiones_ratio"] = df["mcomisiones"] / (1 + np.abs(df["mrentabilidad"]))

    # ===============================
    # 3. CRÉDITO Y APALANCAMIENTO
    # ===============================
    df["deuda_tarjetas_total"] = df["Visa_msaldototal"] + df["Master_msaldototal"]
    df["limite_tarjetas_total"] = df["Visa_mlimitecompra"] + df["Master_mlimitecompra"]
    df["uso_credito_ratio"] = df["deuda_tarjetas_total"] / (1 + df["limite_tarjetas_total"])
    df["mora_total"] = ((df["Visa_delinquency"] == 1) | (df["Master_delinquency"] == 1)).astype(int)
    df["dias_mora_promedio"] = df[["Visa_Finiciomora", "Master_Finiciomora"]].mean(axis=1)

    # ===============================
    # 4. AHORRO / INVERSIÓN
    # ===============================
    df["saldo_total_pesos"] = (
        df["mcaja_ahorro"] + df["mcuenta_corriente"] + df["mplazo_fijo_pesos"] +
        df["minversion1_pesos"] + df["minversion2"]
    )
    df["saldo_total_dolares"] = (
        df["mcaja_ahorro_dolares"] + df["mplazo_fijo_dolares"] + df["minversion1_dolares"]
    )
    df["saldo_total"] = df["saldo_total_pesos"] + df["saldo_total_dolares"]
    df["inversion_ratio"] = (
        (df["mplazo_fijo_pesos"] + df["minversion1_pesos"] + df["minversion2"]) /
        (1 + df["saldo_total"])
    )
    df["diversificacion_financiera"] = (
        (df[["mplazo_fijo_pesos", "mplazo_fijo_dolares", "minversion1_pesos", "minversion2"]] > 0)
        .sum(axis=1)
    )

    # ===============================
    # 5. MOVIMIENTOS OPERATIVOS
    # ===============================
    df["flujo_netotransf"] = df["mtransferencias_recibidas"] - df["mtransferencias_emitidas"]
    df["flujo_cheques"] = df["mcheques_depositados"] - df["mcheques_emitidos"]
    df["saldo_vs_movimientos"] = (
        df["mcuentas_saldo"] / (1 + df["ctransferencias_emitidas"] + df["ctransferencias_recibidas"])
    )
    df["extracciones_ratio"] = df["mextraccion_autoservicio"] / (1 + df["mcuentas_saldo"])

    # ===============================
    # 6. INGRESOS / PAYROLL
    # ===============================
    df["ingresos_mensuales"] = df["mpayroll"] + df["mpayroll2"]
    df["ingresos_vs_saldo"] = df["ingresos_mensuales"] / (1 + df["mcuentas_saldo"])
    df["es_empleado"] = (df["cpayroll_trx"] > 0).astype(int)
    df["cantidad_empleadores"] = df["cpayroll_trx"]

    # ===============================
    # 7. FLAGS BINARIOS / CONDICIONALES
    # ===============================
    df["sin_transacciones_mes"] = (df["actividad_total"] == 0).astype(int)
    df["sin_sueldo_mes"] = (df["ingresos_mensuales"] == 0).astype(int)
    df["alta_mora"] = (df["mora_total"] == 1).astype(int)
    df["sin_inversiones"] = (
        (df["mplazo_fijo_pesos"] + df["minversion1_pesos"] + df["minversion2"] == 0)
    ).astype(int)
    df["solo_tarjetas"] = (
        ((df["ctarjeta_visa"] + df["ctarjeta_master"]) > 0) & (df["tcuentas"] == 0)
    ).astype(int)

    # ===============================
    # 8. INTERACCIONES / RELACIONES
    # ===============================
    df["ratio_productos_uso"] = df["actividad_total"] / (1 + df["cproductos"])
    df["ingresos_por_producto"] = df["ingresos_mensuales"] / (1 + df["cproductos"])
    df["margen_por_cuenta"] = df["margen_total"] / (1 + df["tcuentas"])
    df["clientes_vip_baja_actividad"] = (
        (df["cliente_vip"] == 1) & (df["actividad_total"] < df["actividad_total"].quantile(0.25))
    ).astype(int)

    return df



# def feature_engineering_robust_by_month(df: pd.DataFrame, columnas: list[str] | None = None) -> pd.DataFrame:
#     """
#     Normaliza variables robustamente (mediana / IQR) por 'foto_mes'.
#     Reemplaza las columnas originales, sin generar duplicados.
#     Si IQR=0 o no se puede calcular, mantiene el valor original.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         DataFrame con columna 'foto_mes'.
#     columnas : list[str] | None
#         Columnas a normalizar. Si es None, se aplicará a todas las numéricas excepto 'foto_mes'.

#     Returns
#     -------
#     pd.DataFrame
#         DataFrame con las columnas normalizadas.
#     """

#     if columnas is None:
#         columnas = [c for c in df.select_dtypes(include=[np.number]).columns if c != "foto_mes"]

#     logger.info(f"Normalizando robustamente {len(columnas)} columnas por 'foto_mes' (Pandas)")

#     # 1️⃣ Calcular mediana e IQR por mes
#     stats = df.groupby("foto_mes")[columnas].agg([
#         np.median,
#         lambda x: np.percentile(x, 75) - np.percentile(x, 25)
#     ])
#     stats.columns = [f"{col}_median" if i == 0 else f"{col}_iqr" for col in columnas for i in range(2)]

#     # 2️⃣ Merge al DataFrame original
#     df = df.merge(stats, on="foto_mes", how="left")

#     # 3️⃣ Normalización robusta, preservando valores originales si IQR=0 o NaN
#     for col in columnas:
#         median_col = f"{col}_median"
#         iqr_col = f"{col}_iqr"
#         iqr = df[iqr_col]

#         # Máscara donde se puede normalizar
#         mask = iqr.notna() & (iqr != 0)
#         df.loc[mask, col] = (df.loc[mask, col] - df.loc[mask, median_col]) / iqr[mask]
#         # Valores fuera de mask quedan iguales (preservando original)

#     # 4️⃣ Eliminar columnas temporales
#     tmp_cols = [c for c in df.columns if c.endswith("_median") or c.endswith("_iqr")]
#     df.drop(columns=tmp_cols, inplace=True)

#     logger.info("Normalización robusta completada")

#     return df


logger = logging.getLogger(__name__)

def feature_engineering_robust_by_month_polars(df: pl.DataFrame, columnas: list[str] | None = None) -> pl.DataFrame:
    """
    Normaliza variables robustamente (mediana / IQR) por 'foto_mes' usando Polars.
    Reemplaza las columnas originales.
    Si no se especifican columnas, aplica a todas las numéricas excepto 'foto_mes'.
    """
    if columnas is None:
        columnas = [c for c, dtype in zip(df.columns, df.dtypes) if dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32] and c != "foto_mes"]

    logger.info(f"Normalizando robustamente {len(columnas)} columnas por 'foto_mes' usando Polars")

    # Calcular mediana e IQR por foto_mes
    med_iqr_exprs = []
    for col in columnas:
        med_iqr_exprs.append(
            pl.col(col).median().alias(f"{col}_median")
        )
        med_iqr_exprs.append(
            (pl.col(col).quantile(0.75) - pl.col(col).quantile(0.25)).alias(f"{col}_iqr")
        )

    stats = df.group_by("foto_mes").agg(med_iqr_exprs)

    # Hacer join para tener mediana e IQR por fila
    df = df.join(stats, on="foto_mes", how="left")

    # Normalizar robustamente
    for col in columnas:
        median_col = f"{col}_median"
        iqr_col = f"{col}_iqr"
        # Evitar división por 0 → reemplaza 0 con NaN
        df = df.with_columns(
            ((pl.col(col) - pl.col(median_col)) / pl.when(pl.col(iqr_col) == 0).then(None).otherwise(pl.col(iqr_col)))
            .fill_null(pl.col(col))  # si IQR=0 → deja valor original
            .alias(col)
        )

    # Eliminar columnas temporales
    tmp_cols = [c for c in df.columns if c.endswith("_median") or c.endswith("_iqr")]
    df = df.drop(tmp_cols)

    logger.info("Normalización robusta completada en Polars")

    return df


def ajustar_por_ipc(df: pd.DataFrame, columnas: list[str], columna_mes: str = 'foto_mes') -> pd.DataFrame:
    """
    Ajusta las columnas especificadas del DataFrame dividiéndolas por el IPC correspondiente
    al mes indicado en 'columna_mes'. Reemplaza los valores originales por los ajustados.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con los datos originales (debe contener una columna con el mes, ej. 'foto_mes')
    columnas : list[str]
        Lista de columnas a ajustar por IPC (ejemplo: ['mcuentas_saldo', 'mpayroll'])
    columna_mes : str, default='foto_mes'
        Nombre de la columna en df que indica el mes a usar para ajustar por IPC.

    Returns
    -------
    pd.DataFrame
        DataFrame con las columnas especificadas ajustadas por IPC.
    """

    logger = logging.getLogger(__name__)
    logger.info("Comenzando ajuste por IPC")

    # Cargar IPC nacional
    ipc_df = pd.read_csv("../../../buckets/b1/Compe_02/data/ipc_nacional_fotomes_completado.csv")

    # Merge para asociar el IPC correspondiente a cada fila
    df = df.merge(
        ipc_df[['foto_mes', 'Nivel general']],
        how='left',
        left_on=columna_mes,
        right_on='foto_mes',
        suffixes=('', '_ipc')
    )

    # Ajustar las columnas especificadas (dividir por IPC)
    for col in columnas:
        if col in df.columns:
            df[col] = df[col] / df['Nivel general']
        else:
            logger.warning(f"La columna {col} no existe en el DataFrame y será omitida.")

    logger.info(f"Ajuste por IPC completado. Se ajustaron {len(columnas)} columnas numéricas.")

    # Eliminar columnas auxiliares del merge
    df = df.drop(columns=['foto_mes_ipc', 'Nivel general'], errors='ignore')

    return df



def detectar_grupo_excluido(study_name: str) -> str | None:
    """
    Detecta si el STUDY_NAME indica exclusión de un grupo de variables.
    Ejemplo: "Experimento__sin_tarjetas_consumos" → retorna "tarjetas_consumos"
    """
    match = re.search(r"__sin_(\w+)", study_name)
    if match:
        return match.group(1)
    return None

def detectar_variable_excluida(study_name: str) -> str | None:
    match = re.search(r"__sin_([a-zA-Z0-9_]+)$", study_name)
    if match:
        return match.group(1)
    return None

