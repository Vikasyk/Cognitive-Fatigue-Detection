🧠 ECG-Based Cognitive Fatigue & Stress Detection Using HRV + Machine Learning

A machine-learning project that extracts Heart Rate Variability (HRV) biomarkers from ECG signals (WESAD dataset) and classifies mental states into:

0 → Neutral

1 → Stress

2 → Amusement / Cognitive Load

📌 1. Project Overview

This project analyzes ECG signals to estimate cognitive fatigue and stress using HRV features and Machine Learning.

We use the WESAD dataset, extract ECG data, compute HRV features (SDNN, RMSSD, HR, pNN50), and train a classifier to identify mental states.

The final model achieves ~62% accuracy using clean ECG HRV features.

2. Motivation (Why this project?)

Stress and mental fatigue reduce productivity, affect health, and impair decision-making.

Wearable sensors like ECG provide reliable physiological markers of stress.

HRV (Heart Rate Variability) is a scientifically proven indicator of:

Stress

Autonomic nervous system imbalance

Cognitive workload

Fatigue

This project builds an automatic stress/fatigue detection pipeline using open wearable data.

📌 3. Dataset Used — WESAD (Wearable Stress & Affect Dataset)

15 subjects

Chest sensors: ECG, EMG, EDA, Respiration, Temperature, ACC

Wrist sensors: BVP, GSR, ACC, Temperature

Labeled emotional states:

0 = Baseline (Neutral)

1 = Stress

2 = Amusement

We use only ECG for HRV extraction.

 📌 4. Project Pipeline
WESAD ECG Data  →  HRV Extraction  →  Feature Dataset  →  ML Training  →  Classification Output

5. Features Extracted (HRV Biomarkers)

All features are calculated by our code, not present in the dataset originally.

| Feature             | Meaning                                     |
| ------------------- | ------------------------------------------- |
| **HR (Heart Rate)** | Beats per minute (derived from RR interval) |
| **SDNN**            | Standard deviation of RR intervals          |
| **RMSSD**           | Short-term HRV (sensitive to stress)        |
| **pNN50**           | % of RR interval differences > 50 ms        |
| **RR Interval**     | Time between two R-peaks (in milliseconds)  |


📌 6. Project Folder Structure
📁 project-folder/
│
├── extract_ecg.py                  # Extract ECG from WESAD pkl files
├── extract_hrv_ecg_final.py        # Convert ECG → HRV features (final script)
├── train_model_clean.py            # Train ML model using clean HRV dataset
├── visualize_confusion.py          # Plot confusion matrix
│
├── wesad_raw_ecg.csv               # Raw ECG dataset (created by script)
├── wesad_hrv_features_ecg.csv      # Final HRV feature dataset
│
└── README.md                       # This file


📌 7. How to Run the Project

Step 1 — Install dependencies
pip install numpy pandas matplotlib scikit-learn neurokit2 seaborn


Step 2 — Extract ECG
python extract_ecg.py
👉 Generates wesad_hrv_features_ecg.csv

Step 4 — Train the Machine Learning Model
python train_model_clean.py

Step 5 — Visualize Confusion Matrix
python visualize_confusion.py


📌 8. Model Used

We tested several models, but the best performance was with:

✔ Random Forest Classifier

Handles noisy physiological signals

Good with small datasets

Works well with non-linear HRV relationships

Final accuracy: 62%

📌 9. Results
✔ Accuracy: 62%
✔ Confusion Matrix Interpretation

Class 0 (Neutral) → Best predicted

Class 1 (Stress) → Moderate accuracy

Class 2 (Amusement) → Some misclassification due to signal similarity

Model can successfully detect fatigue/stress patterns from HRV features.



