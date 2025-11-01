import yaml
import subprocess
from datetime import datetime
from config import GRUPOS_VARIABLES

def ejecutar_cmd(cmd):
    """Ejecuta un comando según el entorno."""
    print(f"🖥️ Ejecutando: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# Cargar YAML original
with open("conf.yaml", "r") as f:
    conf_yaml = yaml.safe_load(f)

STUDY_BASE = conf_yaml["STUDY_NAME"]

# Iterar por cada grupo
for grupo in GRUPOS_VARIABLES:
    nuevo_nombre = f"{STUDY_BASE}__sin_{grupo}"
    print(f"\n🔄 Ejecutando experimento: {nuevo_nombre}")

    # Actualizar YAML
    conf_yaml["STUDY_NAME"] = nuevo_nombre
    with open("conf.yaml", "w") as f:
        yaml.dump(conf_yaml, f)

    # Ejecutar pipeline
    ejecutar_cmd("python run_pipeline.py")

print("\n✅ Todos los experimentos finalizados.")

