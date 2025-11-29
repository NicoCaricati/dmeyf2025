from config import *

def analizar_resultados_optuna():
    import optuna
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import json
    import os

    archivo_base = STUDY_NAME

    # Cargar el estudio
    study = optuna.load_study(
        study_name=archivo_base,
        storage=f"sqlite:///../../../buckets/b1/Compe_02/optuna_db/{archivo_base}.db"
    )

    # Crear carpeta
    os.makedirs("resultados", exist_ok=True)

    # Funciones de gráficos
    def plot_histograma_ganancia():
        valores = [t.value for t in study.trials if t.value is not None]
        plt.figure(figsize=(8, 4))
        sns.histplot(valores, bins=20, kde=True)
        plt.title("Distribución de ganancia en validación")
        plt.xlabel("Ganancia")
        plt.ylabel("Frecuencia")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"resultados/{archivo_base}_histograma_ganancia.png")
        plt.close()

    def plot_ganancia_por_trial():
        valores = [t.value for t in study.trials if t.value is not None]
        plt.figure(figsize=(8, 4))
        plt.plot(valores, marker='o')
        plt.title("Ganancia por número de trial")
        plt.xlabel("Trial")
        plt.ylabel("Ganancia")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"resultados/{archivo_base}_ganancia_por_trial.png")
        plt.close()

    # Ejecutar gráficos
    plot_histograma_ganancia()
    plot_ganancia_por_trial()

    # Exportar resultados
    datos_trials = []
    for t in study.trials:
        if t.value is not None:
            fila = {
                "trial_number": t.number,
                "ganancia": t.value,
                **t.params
            }
            datos_trials.append(fila)

    with open(f"resultados/{archivo_base}_iteraciones.json", "w") as f:
        json.dump(datos_trials, f, indent=2)

    pd.DataFrame(datos_trials).to_csv(f"resultados/{archivo_base}_iteraciones.csv", index=False)

    print(f"✅ Resultados guardados en carpeta 'resultados/' con base '{archivo_base}'")
