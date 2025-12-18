import os
import glob
import pickle
import numpy as np
import pandas as pd
import neurokit2 as nk  # pip install neurokit2

# Adjust this pattern to where your WESAD .pkl files are
WESAD_PKL_GLOB = r"C:/Users/Yashaswini Kanthraj/Desktop/bio saftey EL/el/pkl/*.pkl"

OUT_CSV = "wesad_multimodal_features.csv"
WINDOW_SECONDS = 60
FS_CHEST = 700  # WESAD chest sampling rate


def get_scalar(val):
    """Convert NeuroKit outputs (scalar/Series/DataFrame) to float."""
    if isinstance(val, (pd.Series, pd.DataFrame)):
        try:
            val = float(val.iloc[0])
        except Exception:
            return np.nan
    try:
        return float(val)
    except Exception:
        return np.nan


def process_subject(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    subject_id = os.path.basename(pkl_path).split(".")[0]  # e.g. "S2"

    chest = data["signal"]["chest"]
    ecg = chest["ECG"].squeeze()
    eda = chest["EDA"].squeeze()
    resp = chest["Resp"].squeeze() if "Resp" in chest else chest["RESP"].squeeze()
    labels = data["label"].squeeze()

    fs = FS_CHEST
    win_len = WINDOW_SECONDS * fs
    n_samples = len(ecg)
    n_windows = n_samples // win_len

    rows = []

    for w in range(n_windows):
        start = w * win_len
        end = start + win_len

        ecg_win = ecg[start:end]
        eda_win = eda[start:end]
        resp_win = resp[start:end]
        label_win = labels[start:end]

        if len(label_win) == 0:
            continue

        # Majority label for the window
        label = int(pd.Series(label_win).mode()[0])
        if label not in [0, 1, 2]:
            continue

        try:
            # ----- ECG -> HRV -----
            ecg_clean = nk.ecg_clean(ecg_win, sampling_rate=fs)
            _, rpeaks = nk.ecg_peaks(ecg_clean, sampling_rate=fs)

            hrv_time = nk.hrv_time(rpeaks, sampling_rate=fs, show=False)
            hrv_freq = nk.hrv_frequency(rpeaks, sampling_rate=fs, show=False)
            hrv_nl = nk.hrv_nonlinear(rpeaks, sampling_rate=fs, show=False)

            # Time domain
            HR = get_scalar(hrv_time.get("HRV_MeanNN", np.nan))
            SDNN = get_scalar(hrv_time.get("HRV_SDNN", np.nan))
            RMSSD = get_scalar(hrv_time.get("HRV_RMSSD", np.nan))
            pNN50 = get_scalar(hrv_time.get("HRV_pNN50", np.nan))

            # Frequency domain
            LF = get_scalar(hrv_freq.get("HRV_LF", np.nan))
            HF = get_scalar(hrv_freq.get("HRV_HF", np.nan))
            LF_HF = get_scalar(hrv_freq.get("HRV_LFHF", np.nan))
            Total_Power = get_scalar(hrv_freq.get("HRV_TP", np.nan))

            # Non-linear
            SD1 = get_scalar(hrv_nl.get("HRV_SD1", np.nan))
            SD2 = get_scalar(hrv_nl.get("HRV_SD2", np.nan))
            SampEn = get_scalar(hrv_nl.get("HRV_SampEn", np.nan))

            # ----- EDA features -----
            eda_clean = nk.eda_clean(eda_win, sampling_rate=fs)
            eda_signals, _ = nk.eda_process(eda_clean, sampling_rate=fs)
            EDA_mean = float(eda_signals["EDA_Clean"].mean())
            EDA_std = float(eda_signals["EDA_Clean"].std())
            EDA_phasic_mean = float(eda_signals["EDA_Phasic"].mean())
            EDA_tonic_mean = float(eda_signals["EDA_Tonic"].mean())

            # ----- RESP features -----
            resp_clean = nk.rsp_clean(resp_win, sampling_rate=fs)
            resp_signals, _ = nk.rsp_process(resp_clean, sampling_rate=fs)
            RESP_rate_mean = float(resp_signals["RSP_Rate"].mean())
            RESP_amp_mean = float(resp_signals["RSP_Amplitude"].mean())

        except Exception:
            # Skip windows where any processing fails
            continue

        row = {
            "subject_id": subject_id,
            "label": label,
            # ECG HRV
            "HR": HR,
            "SDNN": SDNN,
            "RMSSD": RMSSD,
            "pNN50": pNN50,
            "LF": LF,
            "HF": HF,
            "LF_HF": LF_HF,
            "Total_Power": Total_Power,
            "SD1": SD1,
            "SD2": SD2,
            "SampEn": SampEn,
            # EDA
            "EDA_mean": EDA_mean,
            "EDA_std": EDA_std,
            "EDA_phasic_mean": EDA_phasic_mean,
            "EDA_tonic_mean": EDA_tonic_mean,
            # RESP
            "RESP_rate_mean": RESP_rate_mean,
            "RESP_amp_mean": RESP_amp_mean,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    all_paths = sorted(glob.glob(WESAD_PKL_GLOB))
    print("Found .pkl files:", all_paths)

    if len(all_paths) == 0:
        print("No WESAD .pkl files found. Check WESAD_PKL_GLOB path.")
        return

    all_dfs = []
    for p in all_paths:
        print(f"Processing {p} ...")
        df_subj = process_subject(p)
        if not df_subj.empty:
            all_dfs.append(df_subj)

    if len(all_dfs) == 0:
        print("No data extracted, check paths or parameters.")
        return

    df = pd.concat(all_dfs, ignore_index=True)

    # Keep only main labels
    df = df[df["label"].isin([0, 1, 2])]

    # Ensure all feature columns numeric
    feature_cols = [
        c for c in df.columns if c not in ["subject_id", "label"]
    ]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=feature_cols)

    df.to_csv(OUT_CSV, index=False)
    print(f"Saved multimodal features to {OUT_CSV}, shape = {df.shape}")
    print(df.dtypes)


if __name__ == "__main__":
    main()
