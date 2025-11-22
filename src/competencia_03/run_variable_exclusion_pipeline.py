import yaml
import subprocess
from datetime import datetime
from config import GRUPOS_VARIABLES
import os

# Forzar ejecución desde el directorio del script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def ejecutar_cmd(cmd):
    """Ejecuta un comando y muestra salida en tiempo real."""
    print(f"🖥️ Ejecutando: {cmd}")
    process = subprocess.Popen(cmd, shell=True)
    process.communicate()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)


# Cargar YAML original una sola vez
with open("conf.yaml", "r") as f:
    conf_yaml_original = yaml.safe_load(f)

STUDY_BASE = conf_yaml_original["STUDY_NAME"]

# Iterar por cada grupo y cada variable dentro del grupo
for grupo, variables in GRUPOS_VARIABLES.items():
    for variable in variables:
        nuevo_nombre = f"{STUDY_BASE}__sin_{variable}"
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
