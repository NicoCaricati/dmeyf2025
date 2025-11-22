import os
import shutil
from datetime import datetime
import logging

def crear_snapshot_modelo(study_name: str, archivos_extra: list[str] = None) -> str:
    """
    Crea una carpeta de respaldo con copia de los archivos del modelo actual.

    Args:
        study_name: nombre del estudio o experimento
        archivos_extra: lista de rutas adicionales a copiar

    Returns:
        str: ruta de la carpeta creada
    """
    # Crear carpeta con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    carpeta_modelo = f"../../../buckets/b1/Compe_02/{study_name}"
    os.makedirs(carpeta_modelo, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.info(f"📁 Creando snapshot del modelo en: {carpeta_modelo}")

    # Archivos base a copiar
    archivos = ["main.py", "conf.yaml","config.py","final_training.py","loader.py","optimization.py","test.py","evaluar_meses_test.py" ]
    if archivos_extra:
        archivos.extend(archivos_extra)

    # Copiar archivos si existen
    for archivo in archivos:
        if os.path.exists(archivo):
            destino = os.path.join(carpeta_modelo, os.path.basename(archivo))
            shutil.copy2(archivo, destino)
            logger.debug(f"Copiado: {archivo} → {destino}")
        else:
            logger.warning(f"⚠️ No se encontró el archivo {archivo}, se omite.")

    return carpeta_modelo
