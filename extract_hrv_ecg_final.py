import pandas as pd
import neurokit2 as nk

print("Loading ECG data...")
df = pd.read_csv("wesad_raw_ecg.csv")

sampling_rate = 700
window_size = sampling_rate * 60   # 60-second windows

rows = []

print("\nExtracting HRV from ECG...\n")

for start in range(0, len(df), window_size):
    chunk = df.iloc[start:start + window_size]

    if len(chunk) < window_size:
        continue

    ecg = chunk["ecg"].values

    try:
        signals, info = nk.ecg_process(ecg, sampling_rate=sampling_rate)
        hrv_time = nk.hrv_time(signals, sampling_rate=sampling_rate)

        rows.append({
            "hr": signals["ECG_Rate"].mean(),
            "sdnn": hrv_time["HRV_SDNN"][0],
            "rmssd": hrv_time["HRV_RMSSD"][0],
            "pnn50": hrv_time["HRV_pNN50"][0],
            "label": chunk["label"].mode()[0],
            "subject": chunk["subject"].iloc[0]
        })

    except Exception as e:
        print("Skipping window due to:", e)

hrv_df = pd.DataFrame(rows)
hrv_df.to_csv("wesad_hrv_features_ecg.csv", index=False)

print("\n✔ SUCCESS!")
print("Saved: wesad_hrv_features_ecg.csv")
print("Rows:", len(hrv_df))
