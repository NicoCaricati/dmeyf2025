import yaml
import subprocess
from datetime import datetime
from config import GRUPOS_VARIABLES

def ejecutar_cmd(cmd):
    """Ejecuta un comando según el entorno."""
    print(f"🖥️ Ejecutando: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# Cargar YAML original
with open("config.yaml", "r") as f:
    config_yaml = yaml.safe_load(f)

STUDY_BASE = config_yaml["STUDY_NAME"]

# Iterar por cada grupo
for grupo in grupos_variables:
    nuevo_nombre = f"{STUDY_BASE}__sin_{grupo}"
    print(f"\n🔄 Ejecutando experimento: {nuevo_nombre}")

    # Actualizar YAML
    config_yaml["STUDY_NAME"] = nuevo_nombre
    with open("config.yaml", "w") as f:
        yaml.dump(config_yaml, f)

    # Ejecutar pipeline
    ejecutar_cmd("python run_pipeline.py")

print("\n✅ Todos los experimentos finalizados.")

