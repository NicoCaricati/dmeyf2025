import json
import pandas as pd
import matplotlib.pyplot as plt

# Cargar JSON
with open("resultados\competencia01_iteraciones.json", "r") as f:
    data = json.load(f)

# Pasar a DataFrame
df = pd.json_normalize(data)

# --- 1. Filtrar por lista de trials ---
def seleccionar_trials(df, trial_list):
    subset = df[df["trial_number"].isin(trial_list)]
    cols = ["trial_number", "value"] + [f"params.{c}" for c in data[0]["params"].keys()]
    return subset[cols]

# ejemplo: trials específicos
print(seleccionar_trials(df, [26, 59, 97,71,14]))

# --- 2. Graficar evolución de value ---
plt.figure(figsize=(8,4))
plt.plot(df["trial_number"], df["value"], marker="o")
plt.xlabel("Trial number")
plt.ylabel("Value")
plt.title("Evolución de value")
plt.grid(True)
plt.show()


# entreno un lgmb con los mejores parametros
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

params = {
    'objective': 'binary',
    'metric': 'custom',
    'boosting_type': 'gbdt',
    'first_metric_only': True,
    'boost_from_average': True,
    'feature_pre_filter': False,
    'max_bin': 31,
    'num_leaves': 45,
    'learning_rate': 0.043781,
    'min_data_in_leaf': 965,
    'feature_fraction': 0.283192,
    'bagging_fraction': 0.997500,
    'seed': 555557,
    'verbose': -1
    }   
# Entrenamiento con los mejores parámetros para ver importancia de variables
df = pd.read_csv("data/competencia_fe_sampled.csv")
X = df.drop(columns=['target', 'foto_mes', 'numero_de_cliente'])
y = df['target']

gbm = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'valid'],
        feval=ganancia_lgb_binary,
        num_boost_round=500,
        callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)],  # opcional, logs cada 50 rounds
    )
