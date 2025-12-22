import streamlit as st
import pandas as pd
import joblib
from PIL import Image

from feature_extraction import extract_features


# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Cognitive Stress Detection",
    layout="wide",
    page_icon="🧠",
)


# ================= LOAD MODEL =================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")  # Typical pattern with joblib models and scalers [web:136]


# ================= SIMPLE, HIGH-CONTRAST CSS =================
st.markdown(
    """
    <style>
    body {
        background-color: #e5ecf4;
    }
    .main {
        background-color: #e5ecf4;
    }
    .block-container {
        padding: 2rem 3rem 3rem 3rem;
        max-width: 1120px;
    }

    h1, h2, h3, h4 {
        color: #0f172a;
        font-weight: 700;
    }
    p, li {
        color: #111827;
        font-size: 0.98rem;
    }
    code {
        background-color: #111827;
        color: #f9fafb;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.85rem;
    }

    .hero {
        background-color: #14532d;
        padding: 18px 22px;
        border-radius: 18px;
        color: #f9fafb;
        box-shadow: 0 10px 24px rgba(15,23,42,0.35);
        margin-bottom: 18px;
    }
    .hero-title {
        font-size: 1.7rem;
        font-weight: 750;
        margin-bottom: 4px;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        opacity: 0.96;
    }

    .card {
        background-color: #ffffff;
        padding: 18px 20px;
        border-radius: 16px;
        box-shadow: 0px 4px 14px rgba(15,23,42,0.08);
        margin-bottom: 18px;
        border: 1px solid #d1d5db;
    }

    .metric-card {
        background-color: #f3f4ff;
        padding: 14px 16px;
        border-radius: 12px;
        border: 1px solid #c7d2fe;
    }

    .stButton > button {
        background: linear-gradient(90deg, #2563eb, #22c55e);
        color: #ffffff;
        border-radius: 999px;
        height: 2.7em;
        font-size: 15px;
        border: none;
        padding: 0 24px;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #1d4ed8, #16a34a);
        color: #ffffff;
    }

    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 13px;
        margin-top: 26px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ================= MODE STATE =================
if "mode" not in st.session_state:
    st.session_state.mode = "prediction"   # "prediction", "features", "about"


# ================= TOP BAR (HERO + BUTTONS) =================
top_left, top_right = st.columns([0.7, 0.3])

with top_left:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">🧠 Cognitive Stress Detection</div>
            <div class="hero-subtitle">
                Upload physiological data (<b>ECG, EDA, Respiration</b>) and obtain an instant
                prediction of <b>Stress</b> vs <b>Non‑Stress</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_right:
    st.write("")
    st.write("")
    if st.button("🧪 Prediction"):
        st.session_state.mode = "prediction"
        st.experimental_rerun()
    if st.button("📊 Features"):
        st.session_state.mode = "features"
        st.experimental_rerun()
    if st.button("📚 About"):
        st.session_state.mode = "about"
        st.experimental_rerun()


# ================= PAGE: PREDICTION =================
if st.session_state.mode == "prediction":
    left_col, right_col = st.columns([1.1, 1])

    with left_col:
        st.markdown(
            """
            <div class="card">
                <h3>📂 1. Upload Raw Signal Data</h3>
                <p>
                    Please upload a <b>CSV file</b> containing three columns:<br>
                    <code>ECG</code>, <code>EDA</code>, and <code>RESP</code>.<br><br>
                    Recommended: at least <b>60 seconds</b> of data sampled at a constant rate.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Drag & drop or browse your CSV file",
            type=["csv"],  # Typical Streamlit usage for CSV upload [web:131]
        )

    with right_col:
        st.markdown(
            """
            <div class="card">
                <h3>⚙️ Pipeline overview</h3>
                <ul>
                    <li>Clean signals and segment into 60‑second windows.</li>
                    <li>Extract <b>HRV</b> features from ECG (SDNN, RMSSD, LF/HF, etc.).</li>
                    <li>Extract <b>EDA</b> (skin conductance) and <b>Respiration</b> features.</li>
                    <li>Feed all features into a tuned <b>Random Forest</b> classifier.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if not all(col in df.columns for col in ["ECG", "EDA", "RESP"]):
            st.error("CSV must contain columns named exactly: ECG, EDA, RESP.")
        else:
            st.success("File uploaded successfully. Ready to extract features and predict.")

            col_btn, _ = st.columns([0.4, 0.6])
            with col_btn:
                predict_clicked = st.button("🔍 Run Stress Prediction")

            if predict_clicked:
                with st.spinner("Extracting features and running model..."):
                    try:
                        features_df = extract_features(
                            df["ECG"].values,
                            df["EDA"].values,
                            df["RESP"].values,
                        )

                        X_scaled = scaler.transform(features_df.values)
                        prediction = model.predict(X_scaled)[0]
                        prob = model.predict_proba(X_scaled)[0]

                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown("### 📊 Prediction Result")

                        if prediction == 1:
                            st.error("Stress Detected")
                            conf = prob[1] * 100
                        else:
                            st.success("Non‑Stress Detected")
                            conf = prob[0] * 100

                        # ===== SINGLE METRIC CARD ONLY =====
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <h4>Model Confidence</h4>
                                <h2>{conf:.1f}%</h2>
                                <p> chance that its prediction is correct for this data.</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.markdown("</div>", unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Error during processing: {e}")

    st.markdown(
        """
        <div class="footer">
            <hr>
            Built for educational / research use.<br>
            Powered by: ECG • EDA • Respiration • HRV • Machine Learning
        </div>
        """,
        unsafe_allow_html=True,
    )


# ================= PAGE: FEATURE DETAILS =================
elif st.session_state.mode == "features":
    st.subheader("📊 Feature details")

    st.markdown(
        """
        ### ECG / HRV features

        - Time‑domain: mean RR interval, heart rate (HR), SDNN, RMSSD, pNN50  
        - Frequency‑domain: low‑frequency power (LF), high‑frequency power (HF), LF/HF ratio, total power  
        - Non‑linear: SD1, SD2, entropy‑based indices (e.g., sample entropy)  

        These capture how the time between heartbeats changes with autonomic nervous system activity,
        which is strongly affected by stress and cognitive load.

        ### EDA features

        - Mean skin conductance level (tonic)  
        - Phasic component level (rapid spikes due to stimuli / stress)  

        EDA reflects sympathetic arousal. Under stress, both tonic level and number/intensity of
        phasic responses tend to increase.

        ### Respiration features

        - Average breathing rate  
        - Average respiration amplitude  

        Breathing often becomes faster or shallower under stress, so respiration patterns provide
        additional information beyond ECG and EDA.

        All these features are computed on each 60‑second window and concatenated into a single
        feature vector that is fed to the Random Forest model.
        """
    )


# ================= PAGE: ABOUT THE MODEL =================
else:
    st.subheader("📚 About the model")

    # Confusion matrix image
    try:
        cm_image = Image.open("Figure_1.png")   # or .jpg depending on your file
        st.image(
            cm_image,
            caption="Confusion Matrix – Multimodal RF (Binary)",
            use_column_width=True,
        )
    except Exception as e:
        st.error(f"Could not load confusion-matrix image: {e}")

    st.markdown(
        """
        ### Evaluation metrics (multimodal binary model)

        - **Best CV macro F1:** 0.8467  
        - **Test accuracy:** 0.8988  
        - **Test macro F1:** 0.8651  

        **Classification report**

        - Class 0 (Non‑Stress): precision **0.92**, recall **0.94**, F1 **0.93**, support **249**  
        - Class 1 (Stress): precision **0.83**, recall **0.77**, F1 **0.80**, support **87**  

        Overall: accuracy **0.90**, macro‑average F1 **0.87**, weighted‑average F1 **0.90**.
        """
    )

    st.markdown(
        """
        ### Interpreting the confusion matrix

        - **235** Non‑Stress windows correctly predicted as Non‑Stress.  
        - **14** Non‑Stress windows predicted as Stress (false positives).  
        - **20** Stress windows predicted as Non‑Stress (missed stress).  
        - **67** Stress windows correctly predicted as Stress.  

        Most samples lie on the diagonal (235 + 67), and off‑diagonal counts (14 + 20) are low,
        which shows that the multimodal Random Forest is reliably distinguishing between
        relaxed and stressed windows.
        """
    )
