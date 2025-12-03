import pickle
import pandas as pd
import numpy as np
import os

DATA_PATH = r"C:/Users/Yashaswini Kanthraj/Desktop/bio saftey/el/pkl/"

subjects = ["S2","S3","S4","S5","S6","S7","S8","S9","S10","S11","S13","S14","S15","S16","S17"]

all_rows = []

print("\nStarting ECG extraction...\n")

for subj in subjects:
    pkl_path = os.path.join(DATA_PATH, f"{subj}.pkl")
    print("Loading:", pkl_path)

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding="latin1")

    ecg = data["signal"]["chest"]["ECG"][:, 0]     # ECG 1D array
    labels = data["label"]

    min_len = min(len(ecg), len(labels))
    
    df = pd.DataFrame({
        "subject": subj,
        "ecg": ecg[:min_len],
        "label": labels[:min_len]
    })

    all_rows.append(df)

final = pd.concat(all_rows, ignore_index=True)
final.to_csv("wesad_raw_ecg.csv", index=False)

print("\n✔ Saved: wesad_raw_ecg.csv")
print("Shape:", final.shape)
