import numpy as np
import pandas as pd
import neurokit2 as nk

FS = 700
WINDOW_SECONDS = 60

def extract_features(ecg, eda, resp):
    ecg = np.array(ecg)
    eda = np.array(eda)
    resp = np.array(resp)

    if len(ecg) < FS * WINDOW_SECONDS:
        raise ValueError("At least 60 seconds of data required")

    ecg_clean = nk.ecg_clean(ecg, sampling_rate=FS)
    _, rpeaks = nk.ecg_peaks(ecg_clean, sampling_rate=FS)

    hrv_time = nk.hrv_time(rpeaks, sampling_rate=FS, show=False)
    hrv_freq = nk.hrv_frequency(rpeaks, sampling_rate=FS, show=False)
    hrv_nl = nk.hrv_nonlinear(rpeaks, sampling_rate=FS, show=False)

    eda_clean = nk.eda_clean(eda, sampling_rate=FS)
    eda_sig, _ = nk.eda_process(eda_clean, sampling_rate=FS)

    resp_clean = nk.rsp_clean(resp, sampling_rate=FS)
    resp_sig, _ = nk.rsp_process(resp_clean, sampling_rate=FS)

    features = {
        "HR": hrv_time["HRV_MeanNN"].iloc[0],
        "SDNN": hrv_time["HRV_SDNN"].iloc[0],
        "RMSSD": hrv_time["HRV_RMSSD"].iloc[0],
        "pNN50": hrv_time["HRV_pNN50"].iloc[0],
        "LF": hrv_freq["HRV_LF"].iloc[0],
        "HF": hrv_freq["HRV_HF"].iloc[0],
        "LF_HF": hrv_freq["HRV_LFHF"].iloc[0],
        "Total_Power": hrv_freq["HRV_TP"].iloc[0],
        "SD1": hrv_nl["HRV_SD1"].iloc[0],
        "SD2": hrv_nl["HRV_SD2"].iloc[0],
        "SampEn": hrv_nl["HRV_SampEn"].iloc[0],
        "EDA_mean": eda_sig["EDA_Clean"].mean(),
        "EDA_std": eda_sig["EDA_Clean"].std(),
        "EDA_phasic_mean": eda_sig["EDA_Phasic"].mean(),
        "EDA_tonic_mean": eda_sig["EDA_Tonic"].mean(),
        "RESP_rate_mean": resp_sig["RSP_Rate"].mean(),
        "RESP_amp_mean": resp_sig["RSP_Amplitude"].mean(),
    }

    return pd.DataFrame([features])
