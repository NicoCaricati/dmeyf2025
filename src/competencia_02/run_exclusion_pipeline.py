import yaml
import subprocess
from datetime import datetime
from config import GRUPOS_VARIABLES
import os

# Forzar ejecución desde el directorio del script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def ejecutar_cmd(cmd):
    """Ejecuta un comando y muestra su salida completa."""
    print(f"🖥️ Ejecutando: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


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
