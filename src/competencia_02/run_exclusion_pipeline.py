import yaml
import subprocess
from datetime import datetime
from config import GRUPOS_VARIABLES
import os

print("Grupos variables cargados:", GRUPOS_VARIABLES)
print("Tipo de GRUPOS_VARIABLES:", type(GRUPOS_VARIABLES))


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

# Cargar YAML original una sola vez
with open("conf.yaml", "r") as f:
    conf_yaml_original = yaml.safe_load(f)

STUDY_BASE = conf_yaml_original["STUDY_NAME"]

# Iterar por cada grupo
for grupo in GRUPOS_VARIABLES:
    nuevo_nombre = f"{STUDY_BASE}__sin_{grupo}"
    print(f"\n🔄 Ejecutando experimento: {nuevo_nombre}")

    # Crear copia limpia del YAML original en cada iteración
    conf_yaml = conf_yaml_original.copy()
    conf_yaml["STUDY_NAME"] = nuevo_nombre

    # Guardar YAML actualizado
    with open("conf.yaml", "w") as f:
        yaml.dump(conf_yaml, f)

    # Ejecutar pipeline
    try:
        ejecutar_cmd("python run_pipeline.py")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en experimento {nuevo_nombre}: {e}")
        continue


print("\n✅ Todos los experimentos finalizados.")
