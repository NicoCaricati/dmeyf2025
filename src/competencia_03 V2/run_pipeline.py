import os
import subprocess
from datetime import datetime

# --- 1. Detección del entorno ---
def en_notebook():
    """Devuelve True si se está ejecutando en un notebook (Jupyter/IPython)."""
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell in ["ZMQInteractiveShell", "Shell"]
    except Exception:
        return False

# --- 2. Ejecutar un comando del sistema ---
def ejecutar_cmd(cmd):
    """Ejecuta un comando según el entorno: notebook usa !python, terminal usa subprocess."""
    if en_notebook():
        from IPython import get_ipython
        print(f"💻 Ejecutando (Jupyter): !{cmd}")
        get_ipython().system(cmd)
    else:
        print(f"🖥️ Ejecutando (terminal): {cmd}")
        subprocess.run(cmd, shell=True, check=True)

# --- 3. Ejecutar pipeline principal ---
def main():
    print("🚀 Iniciando pipeline...")

    # Asegurar carpetas necesarias
    for carpeta in ["logs", "output", "resultados", "feature_importance", "predict"]:
        os.makedirs(carpeta, exist_ok=True)

    # --- Ejecutar main.py garantizado ---
    ejecutar_cmd("python main.py")

    # --- Subir resultados a Git ---
    try:
        ejecutar_cmd("git push")
    except subprocess.CalledProcessError as e:
        if "no upstream branch" in str(e):
            # Configurar upstream si falta
            result = subprocess.run("git branch --show-current", shell=True, capture_output=True, text=True)
            branch = result.stdout.strip()
            print(f"⚙️ Configurando upstream para la rama '{branch}'...")
            ejecutar_cmd(f"git push --set-upstream origin {branch}")
        else:
            raise

    # Obtener STUDY_NAME desde config.py
    try:
        result = subprocess.run(
            ["python", "-c", "from config import STUDY_NAME; print(STUDY_NAME)"],
            capture_output=True, text=True, check=True
        )
        STUDY_NAME = result.stdout.strip()
    except subprocess.CalledProcessError:
        STUDY_NAME = "UNKNOWN"

    print(f"📂 Subiendo resultados para estudio: {STUDY_NAME}")

    # Agregar cambios a Git (forzar si están en .gitignore)
    ejecutar_cmd("git add -f logs/ predict/ resultados/ feature_importance/ || true")

    # Hacer commit y push si hay cambios
    diff_check = subprocess.run("git diff --cached --quiet", shell=True)
    if diff_check.returncode != 0:
        mensaje = f"Resultados automáticos [{STUDY_NAME}] {datetime.now():%Y-%m-%d %H:%M}"
        ejecutar_cmd(f'git commit -m "{mensaje}"')
        ejecutar_cmd("git push")
        print("✅ Resultados subidos correctamente.")
    else:
        print("⚠️ No hay cambios para commitear.")

    print("🏁 Pipeline finalizado con éxito.")

# --- 4. Llamar a main() si es script principal ---
if __name__ == "__main__":
    main()
