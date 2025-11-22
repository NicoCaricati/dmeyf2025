import duckdb

path_02 = "../../../buckets/b1/datasets/competencia_02_crudo.csv.gz"
path_03 = "../../../buckets/b1/datasets/competencia_03_crudo.csv.gz"
path_04 = "../../../buckets/b1/datasets/competencia_04_crudo.csv.gz"

con = duckdb.connect()

# Concatenar directamente: 02 con header, 03 sin header (saltando la primera fila)
con.execute(f"""
    COPY (
        SELECT * FROM read_csv_auto('{path_02}', header=True, compression='gzip')
        UNION ALL
        SELECT * FROM read_csv_auto('{path_03}', header=False, skip=1, compression='gzip')
    ) TO '{path_04}' (FORMAT CSV, COMPRESSION GZIP, HEADER TRUE)
""")

con.close()
