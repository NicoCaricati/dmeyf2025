# src/config.py
import yaml
import os
import logging

logger = logging.getLogger(__name__)

#Ruta del archivo de configuracion
PATH_CONFIG = os.path.join(os.path.dirname(__file__), "conf.yaml")

try:
    with open(PATH_CONFIG, "r") as f:
        _cfgGeneral = yaml.safe_load(f)
        _cfg = _cfgGeneral["competencia01"]
        STUDY_NAME = _cfgGeneral.get("STUDY_NAME", "Wednesday")
        DATA_PATH = _cfg.get("DATA_PATH", "../data/competencia_03.csv")
        BUCKET_NAME = _cfg.get("BUCKET_NAME", "../../../buckets/b1/Compe_02")
        SEMILLA = _cfg.get("SEMILLA", [42])
        MES_TRAIN = _cfg.get("MES_TRAIN",[202101,202102,202103])
        MES_VALIDACION = _cfg.get("MES_VALIDACION", 202104)
        MES_TEST = _cfg.get("MES_TEST", 202106)
        MESES_OPTIMIZACION =  _cfg.get("MESES_OPTIMIZACION", [202101,202102,202103,202104])
        GANANCIA_ACIERTO = _cfg.get("GANANCIA_ACIERTO", None)
        COSTO_ESTIMULO = _cfg.get("COSTO_ESTIMULO", None)
        FINAL_TRAIN = _cfg.get("FINAL_TRAIN", [202101, 202102, 202103, 202104])
        FINAL_PREDIC = _cfg.get("FINAL_PREDIC", 202106)
        UMBRAL = _cfg.get("UMBRAL", 0.04)
        HYPERPARAM_RANGES = _cfg.get("HYPERPARAM_RANGES", {})
        TOP_K = _cfg.get("TOP_K", 10000)
        UNDERSAMPLING = cfg.get("UNDERSAMPLING", 0.2)
        MESES_EVALUACION = _cfg.get("MESES_EVALUACION", {})
        if not isinstance(MESES_EVALUACION, dict):
            raise ValueError("La sección MESES_EVALUACION debe ser un diccionario en el YAML.")
        GRUPOS_VARIABLES = _cfg.get("GRUPOS_VARIABLES", {})




except Exception as e:
    logger.error(f"Error al cargar el archivo de configuracion: {e}")
    raise