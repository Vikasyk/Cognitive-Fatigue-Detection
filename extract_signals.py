# ==========================================================
# Cognitive Fatigue Forecasting
# FINAL Correct Signal Extraction Script
# ==========================================================

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk

# ----------------------------------------------------------
# PATHS
# ----------------------------------------------------------
DATASET_DIR = r"C:\Users\YK\Desktop\bio saftey\pkl"
DATASET_FILE = os.path.join(DATASET_DIR, "S2.pkl")
OUTPUT_DIR = os.path.join(DATASET_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------
# LOAD DATASET (Python2 → Python3 pickle fix)
# ----------------------------------------------------------
with open(DATASET_FILE, "rb") as f:
    data = pickle.load(f, encoding="latin1")

print("✅ Dataset loaded")
print("Dataset keys:", data.keys())
print("Signal keys:", data["signal"].keys())

# ----------------------------------------------------------
# EXTRACT SIGNALS (CORRECT STRUCTURE)
# ----------------------------------------------------------
chest = data["signal"]["chest"]
wrist = data["signal"]["wrist"]

# Extract signals
ecg_signal  = np.array(chest["ECG"]).flatten()
resp_signal = np.array(chest["Resp"]).flatten()
eda_signal  = np.array(wrist["EDA"]).flatten()

# Sampling rate
fs = 700

print("Sampling rate:", fs)
print("ECG length:", len(ecg_signal))
print("EDA length:", len(eda_signal))
print("RESP length:", len(resp_signal))

# ----------------------------------------------------------
# PLOTTING FUNCTION
# ----------------------------------------------------------
def plot_signal(sig, title, color, filename, ylabel):
    plt.figure(figsize=(10, 3))
    plt.plot(sig[:5000], color=color)
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.show()

# ----------------------------------------------------------
# PLOT RAW SIGNALS (FOR PAPER)
# ----------------------------------------------------------
plot_signal(ecg_signal,  "ECG Signal",         "blue",   "ecg_signal.png",  "Amplitude")
plot_signal(eda_signal,  "EDA Signal",         "orange", "eda_signal.png",  "Skin Conductance")
plot_signal(resp_signal, "Respiration Signal", "green",  "resp_signal.png", "Amplitude")

# ----------------------------------------------------------
# ECG PROCESSING → RR INTERVALS
# ----------------------------------------------------------
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=fs)
_, ecg_info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs)

r_peaks = ecg_info["ECG_R_Peaks"]
rr_intervals = np.diff(r_peaks) / fs

print("Mean RR interval:", np.mean(rr_intervals))

# ----------------------------------------------------------
# EDA PROCESSING → TONIC & PHASIC
# ----------------------------------------------------------
eda_signals, _ = nk.eda_process(eda_signal, sampling_rate=fs)
eda_tonic  = eda_signals["EDA_Tonic"]
eda_phasic = eda_signals["EDA_Phasic"]

# ----------------------------------------------------------
# RESPIRATION PROCESSING
# ----------------------------------------------------------
_, resp_info = nk.rsp_process(resp_signal, sampling_rate=fs)
resp_rate = resp_info["RSP_Rate"]

print("Mean Respiration Rate:", np.nanmean(resp_rate))

# ----------------------------------------------------------
# SAVE OUTPUTS
# ----------------------------------------------------------
np.save(os.path.join(OUTPUT_DIR, "ecg_signal.npy"), ecg_signal)
np.save(os.path.join(OUTPUT_DIR, "eda_signal.npy"), eda_signal)
np.save(os.path.join(OUTPUT_DIR, "resp_signal.npy"), resp_signal)
np.save(os.path.join(OUTPUT_DIR, "rr_intervals.npy"), rr_intervals)

print("✅ Signal extraction completed successfully")
print("📁 Outputs saved in:", OUTPUT_DIR)
