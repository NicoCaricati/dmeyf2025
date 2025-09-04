# No hacer scripts sueltos de cada tema. Ir integrando todo en un solo script.
# Dejar todo en un archivo bastante limpio y ordenado.
# Hasta ahora agregar clase ternaria, arboles, bosques, obstuza
import pandas as pd
import numpy as np
import os

def main():
    print("Inicio de Ejecucion")

    # Carga de datos
    try:
        df = pd.read_csv("data/competencia_01.csv")
        print(df.head())
        print(df.columns)
        print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
        print("Dataset Cargado")
    except FileNotFoundError:
        print("Error: El archivo 'competencia_01.csv' no se encuentra.")
        return



if __name__ == "__main__":
    main()
    


print("Hola Mundo")