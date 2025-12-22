import pickle
import numpy as np
import pandas as pd

# ========== CONFIG ==========
PKL_PATH = "pkl/S2.pkl"        # path to S2.pkl
FS = 700                       # sampling rate
SECONDS = 60                  # duration
OUT_CSV = "sample_input.csv"

# ========== LOAD PKL ==========
with open(PKL_PATH, "rb") as f:
    data = pickle.load(f, encoding="latin1")

# ========== EXTRACT SIGNALS ==========
ecg = data["signal"]["chest"]["ECG"].squeeze()
eda = data["signal"]["chest"]["EDA"].squeeze()
resp = data["signal"]["chest"]["Resp"].squeeze()

# ========== TAKE FIRST 60 SECONDS ==========
n_samples = FS * SECONDS

ecg_60 = ecg[:n_samples]
eda_60 = eda[:n_samples]
resp_60 = resp[:n_samples]

# ========== CREATE CSV ==========
df = pd.DataFrame({
    "ECG": ecg_60,
    "EDA": eda_60,
    "RESP": resp_60
})

df.to_csv(OUT_CSV, index=False)

# ========== INFO ==========
print("✅ sample_input.csv created successfully")
print("Rows:", df.shape[0])
print("Columns:", df.columns.tolist())
print(df.head())
