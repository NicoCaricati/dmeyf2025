import pandas as pd
import duckdb
import logging
import numpy as np
from itertools import combinations


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
    sql = "SELECT *"
  
    # Agregar los lags para los atributos especificados
    for attr in columnas:
        if attr in df.columns:
            for i in range(1, cant_lag + 1):
                sql += f", lag({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_lag_{i}"
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
    sql = "SELECT *"
  
    # Agregar los deltas para los atributos especificados
    for attr in columnas:
        if attr in df.columns:
            for i in range(1, cant_delta + 1):
                sql += f", {attr} - LAG({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_delta_{i}"
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

def feature_engineering_tasa_variacion(df: pd.DataFrame, columnas: list[str], cant_tasa_variacion: int = 1) -> pd.DataFrame:
    """
    Genera variables de tasa de variación para los atributos especificados utilizando SQL.
    La tasa se define como (X_t - X_{t-i}) / X_{t-i}.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list[str]
        Lista de atributos para los cuales generar tasas de variación
    cant_delta : int, default=1
        Cantidad de lags a usar para calcular tasas
    
    Returns
    -------
    pd.DataFrame
        DataFrame con las tasas de variación agregadas
    """
    if not columnas:
        logger.warning("No se especificaron atributos para generar tasas de variación")
        return df

    sql = "SELECT *,\n"

    exprs = []
    for attr in columnas:
        if attr in df.columns:
            for i in range(1, cant_tasa_variacion + 1):
                exprs.append(
                    f"CASE WHEN LAG({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) IS NULL "
                    f"OR LAG({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) = 0 "
                    f"THEN NULL ELSE "
                    f"(({attr} - LAG({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes)) "
                    f"/ NULLIF(LAG({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes),0.0)) END "
                    f"AS {attr}_tasa_var_{i}"
                )
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    sql += ",\n".join(exprs)
    sql += " FROM df"

    logger.debug(f"Consulta SQL: {sql}")

    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df()
    con.close()

    logger.info(f"Tasas de variación generadas. DataFrame resultante con {df.shape[1]} columnas")

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

    import numpy as np

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
                    ) AS slope_{col}_window
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

            col_name_inv = f"ratio_{denominador}_{numerador}"
            df[col_name_inv] = df[denominador] / df[numerador].replace(0, np.nan)
        
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

        sum(
            coalesce(ctarjeta_debito_transacciones,0)
            + coalesce(ctarjeta_visa_transacciones,0)
            + coalesce(ctarjeta_master_transacciones,0)
            + coalesce(ccajas_transacciones,0)
        ) OVER (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS ctrx_120d

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


