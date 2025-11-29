# evaluar_meses_test.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from test import comparar_semillas_en_grafico_con_ensamble
from config import BUCKET_NAME, UMBRAL
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def evaluar_meses_test(df_fe, mejores_params, semillas, study_name, config_meses):
    resumen = []
    curvas_por_mes = {}

    for mes_test, config in config_meses.items():
        logger.info(f"📅 Evaluando mes de test: {mes_test}")
        train_periodos = config["train"]

        # Seteo global temporal
        global MES_TEST, MES_TRAIN
        MES_TEST = int(mes_test)
        MES_TRAIN = train_periodos

        # Nombre único para este test
        nombre_archivo  = f"{study_name}_test_{mes_test}"

        # Ejecutar evaluación
        resultados = comparar_semillas_en_grafico_con_ensamble(
            df_fe=df_fe,
            mejores_params=mejores_params,
            semillas=semillas,
            mes_test=int(mes_test),
            meses_train=train_periodos,
            study_name=study_name)


        res = resultados["resultados_ensamble"]
        curva_prom = resultados["curva_prom"]
        curva_std = resultados["curva_std"]

        curvas_por_mes[mes_test] = curva_prom

        resumen.append({
            "study_name": study_name,
            "foto_mes_test": mes_test,
            "ganancia_test": res["ganancia_test"],
            "predicciones_positivas": res["predicciones_positivas"],
            "porcentaje_positivas": res["porcentaje_positivas"],
            "ganancia_optima_promedio": float(max(curva_prom)),
            "clientes_optimos": int(curva_prom.argmax())
        })

    # Guardar CSV resumen
    df_resumen = pd.DataFrame(resumen)
    os.makedirs(f"{BUCKET_NAME}/{study_name}", exist_ok=True)
    ruta_csv = f"{BUCKET_NAME}/{study_name}/resumen_evaluacion_multimes.csv"
    df_resumen.to_csv(ruta_csv, index=False)
    logger.info(f"📄 CSV resumen guardado: {ruta_csv}")

    # Graficar curvas combinadas
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for mes, curva in curvas_por_mes.items():
        ax.plot(curva, lw=2.5, label=f"{mes} (Ganancia: {max(curva):,.0f})")

    ax.set_title(f"Comparativo de curvas promedio por mes - {study_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Clientes ordenados por probabilidad", fontsize=12)
    ax.set_ylabel("Ganancia acumulada", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))
    ax.set_xlim(right=20000)
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_plot = f"{BUCKET_NAME}/{study_name}/comparativo_multimes_{timestamp}.png"
    plt.savefig(ruta_plot, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"📊 Gráfico combinado guardado: {ruta_plot}")
