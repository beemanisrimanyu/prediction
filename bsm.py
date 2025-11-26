# app_streamlit_public.py
"""
Public-ready Streamlit dashboard for rGO + UFS concrete compressive strength prediction.

Behavior:
- Try to download model+scalers from URLs provided by environment variables:
    MODEL_URL, SCALERX_URL, SCALERY_URL
  (These should point to raw files accessible without auth, e.g. GitHub raw or S3 public URLs.)
- If URLs are not provided or download fails, fallback to local files:
    mlp_rgo_ufs_strength.joblib, scalerX_rgo_ufs.joblib, scalery_rgo_ufs.joblib
- Basic per-session rate-limiting (max predictions per minute).
- Batch upload (CSV/XLSX) and single sample inputs supported.
- Saves a small prediction log file: predictions.log
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import requests
import time
from io import BytesIO, StringIO
from pathlib import Path

st.set_page_config(page_title="Public: rGO+UFS Concrete Predictor", layout="wide")

# -------------------------
# Configuration
# -------------------------
FEATURES = ['rGO_%', 'UFS_%', 'w_b_ratio', 'Age_days']
LOCAL_MODEL_FILES = {
    "model": "mlp_rgo_ufs_strength.joblib",
    "scalerX": "scalerX_rgo_ufs.joblib",
    "scalery": "scalery_rgo_ufs.joblib"
}
# environment variable names to fetch model files from (public links)
ENV_MODEL_URL = "MODEL_URL"
ENV_SCALERX_URL = "SCALERX_URL"
ENV_SCALERY_URL = "SCALERY_URL"

# rate limit settings (per session)
MAX_PREDICTIONS_PER_MINUTE = 40

LOG_FILE = "predictions.log"

# -------------------------
# Utility functions
# -------------------------
def log_event(event_type, msg):
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(LOG_FILE, "a") as f:
        f.write(f"{t}\t{event_type}\t{msg}\n")

@st.cache_data(show_spinner=False)
def download_file(url, dest_path, timeout=30):
    """Download URL to dest_path. Returns True if successful."""
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(r.content)
        return True, None
    except Exception as e:
        return False, str(e)

def load_artifacts():
    """
    Try to load model & scalers from env URLs if present, otherwise try local files.
    Returns (model, scalerX, scalerY, messages_list)
    """
    msgs = []
    # Try environment URLs first
    model_url = os.getenv(ENV_MODEL_URL)
    scalerx_url = os.getenv(ENV_SCALERX_URL)
    scalery_url = os.getenv(ENV_SCALERY_URL)

    artifact_paths = {}
    # Temporary local filenames
    tmp_dir = Path("artifacts")
    tmp_dir.mkdir(exist_ok=True)
    if model_url and scalerx_url and scalery_url:
        # attempt download
        for name, url in [("model", model_url), ("scalerX", scalerx_url), ("scalery", scalery_url)]:
            dest = tmp_dir / f"{name}.joblib"
            ok, err = download_file(url, dest)
            if ok:
                artifact_paths[name] = str(dest)
                msgs.append(f"Downloaded {name} from env URL.")
            else:
                msgs.append(f"Failed to download {name} from URL: {err}")
                artifact_paths = {}
                break

    # If no env URLs or download failed, try local files
    if not artifact_paths:
        missing = []
        for key, fname in LOCAL_MODEL_FILES.items():
            if Path(fname).exists():
                artifact_paths[key] = fname
            else:
                missing.append(fname)
        if missing:
            msgs.append(f"Missing local files: {missing}. Provide them or set environment URLs.")
        else:
            msgs.append("Loaded artifacts from local files.")

    # Load them if we have paths
    if set(["model", "scalerX", "scalery"]).issubset(artifact_paths.keys()):
        try:
            model = joblib.load(artifact_paths["model"])
            scalerX = joblib.load(artifact_paths["scalerX"])
            scalerY = joblib.load(artifact_paths["scalery"])
            msgs.append("Artifacts loaded successfully.")
            return model, scalerX, scalerY, msgs
        except Exception as e:
            msgs.append(f"Failed to load artifacts: {e}")
            return None, None, None, msgs
    else:
        return None, None, None, msgs

def validate_input_row(row):
    """Return (ok, msg). row is dict or Series with FEATURES keys."""
    try:
        rgo = float(row['rGO_%'])
        ufs = float(row['UFS_%'])
        wb = float(row['w_b_ratio'])
        age = float(row['Age_days'])
        if not (0 <= rgo <= 5):
            return False, "rGO_% should be in [0, 5] (use realistic percent)."
        if not (0 <= ufs <= 100):
            return False, "UFS_% should be in [0, 100]."
        if not (0.1 <= wb <= 1.0):
            return False, "w_b_ratio should be in [0.1, 1.0]."
        if not (1 <= age <= 3650):
            return False, "Age_days should be between 1 and 3650."
        return True, ""
    except Exception as e:
        return False, f"Invalid input: {e}"

def predict_df(df_in, model, scalerX, scalerY):
    X = df_in[FEATURES].astype(float).values
    Xs = scalerX.transform(X)
    preds_s = model.predict(Xs)
    preds = scalerY.inverse_transform(preds_s.reshape(-1,1)).ravel()
    out = df_in.copy()
    out["Predicted_Compressive_Strength_MPa"] = np.round(preds, 3)
    return out

def to_excel_bytes(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="predictions")
        writer.save()
    return output.getvalue()

# -------------------------
# Load artifacts (cached)
# -------------------------
with st.spinner("Loading model artifacts..."):
    MODEL, SCALER_X, SCALER_Y, LOAD_MSGS = load_artifacts()

# -------------------------
# Sidebar (info + rate-limits)
# -------------------------
st.sidebar.title("rGO + UFS Concrete Predictor (Public)")
st.sidebar.info("This demo provides approximate predictions. Validate experimentally before use.")
for m in LOAD_MSGS:
    st.sidebar.write("- " + m)

st.sidebar.markdown("---")
st.sidebar.subheader("Usage & Limits")
st.sidebar.write(f"- Max predictions / session-minute: **{MAX_PREDICTIONS_PER_MINUTE}** (per session)")
st.sidebar.write("- Batch uploads limited by Streamlit plan; keep batch sizes moderate.")
st.sidebar.markdown("---")
st.sidebar.write("If you are the app owner, set environment variables MODEL_URL, SCALERX_URL, SCALERY_URL to public raw links to serve model files from cloud storage.")

# -------------------------
# Initialize per-session counters
# -------------------------
if "pred_timestamps" not in st.session_state:
    st.session_state.pred_timestamps = []  # list of UNIX timestamps for prediction calls

def check_rate_limit():
    now = time.time()
    window_start = now - 60  # 60 seconds window
    # remove timestamps older than window
    st.session_state.pred_timestamps = [t for t in st.session_state.pred_timestamps if t >= window_start]
    if len(st.session_state.pred_timestamps) >= MAX_PREDICTIONS_PER_MINUTE:
        return False, f"Rate limit exceeded: {len(st.session_state.pred_timestamps)} predictions in last minute."
    # allow
    st.session_state.pred_timestamps.append(now)
    return True, ""

# -------------------------
# Main UI
# -------------------------
col1, col2 = st.columns([1,2])
with col1:
    st.header("Single-sample prediction")
    rgo = st.number_input("rGO (%)", min_value=0.0, max_value=5.0, value=0.05, step=0.01, format="%.3f")
    ufs = st.number_input("UFS (%)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
    wbr = st.number_input("w/b ratio", min_value=0.1, max_value=1.0, value=0.35, step=0.01, format="%.2f")
    age = st.number_input("Age (days)", min_value=1, max_value=3650, value=28, step=1)

    if st.button("Predict single sample"):
        if MODEL is None:
            st.error("Model not available. Check app owner configuration or place model files locally.")
            log_event("ERROR", "Single prediction attempted but model missing.")
        else:
            ok_rate, msg_rate = check_rate_limit()
            if not ok_rate:
                st.warning(msg_rate)
            else:
                row = {'rGO_%': rgo, 'UFS_%': ufs, 'w_b_ratio': wbr, 'Age_days': age}
                ok, msg = validate_input_row(row)
                if not ok:
                    st.error("Invalid input: " + msg)
                else:
                    df_in = pd.DataFrame([row])[FEATURES]
                    try:
                        out = predict_df(df_in, MODEL, SCALER_X, SCALER_Y)
                        st.success(f"Predicted compressive strength: {out['Predicted_Compressive_Strength_MPa'].iloc[0]:.2f} MPa")
                        st.table(out)
                        log_event("PREDICT_SINGLE", f"{row} => {out['Predicted_Compressive_Strength_MPa'].iloc[0]:.3f}")
                    except Exception as e:
                        st.error("Prediction failed: " + str(e))
                        log_event("ERROR", f"Single predict failed: {e}")

    st.markdown("---")
    st.header("Batch prediction (CSV / XLSX)")
    uploaded = st.file_uploader("Upload CSV or XLSX with columns: rGO_%, UFS_%, w_b_ratio, Age_days", type=["csv","xlsx"])
    st.caption("Optionally include Compressive_Strength_MPa column to compare actual vs predicted.")

    st.markdown("### Example rows (copy to CSV)")
    example = pd.DataFrame([
        [0.00, 0, 0.35, 7],
        [0.05, 10, 0.35, 28],
        [0.10, 10, 0.35, 90],
        [0.02, 20, 0.40, 28],
        [0.10, 30, 0.35, 56],
    ], columns=FEATURES)
    st.dataframe(example, height=170)

with col2:
    st.header("Results & Visuals")
    if uploaded is not None:
        if MODEL is None:
            st.error("Model not available. Batch prediction not possible.")
        else:
            # rate limit check (count the whole batch as 1 call but still limit)
            ok_rate, msg_rate = check_rate_limit()
            if not ok_rate:
                st.warning(msg_rate)
            else:
                try:
                    if uploaded.name.endswith(".xlsx"):
                        df_up = pd.read_excel(uploaded)
                    else:
                        df_up = pd.read_csv(uploaded)
                except Exception as e:
                    st.error("Failed to read uploaded file: " + str(e))
                    df_up = None

                if df_up is not None:
                    # check needed columns
                    missing = [c for c in FEATURES if c not in df_up.columns]
                    if missing:
                        st.error(f"Uploaded file missing columns: {missing}")
                    else:
                        # validate each row
                        bad_rows = []
                        for i, row in df_up.iterrows():
                            ok, msg = validate_input_row(row)
                            if not ok:
                                bad_rows.append((i, msg))
                        if bad_rows:
                            st.error(f"{len(bad_rows)} rows invalid. First errors: {bad_rows[:5]}")
                        else:
                            try:
                                out_df = predict_df(df_up, MODEL, SCALER_X, SCALER_Y)
                                st.success(f"Predicted {len(out_df)} rows.")
                                st.dataframe(out_df.head(200))
                                # download
                                bytes_xl = to_excel_bytes(out_df)
                                st.download_button("Download predictions (.xlsx)", data=bytes_xl, file_name="predictions.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                                # simple visuals
                                if "Compressive_Strength_MPa" in out_df.columns:
                                    st.markdown("#### Predicted vs Actual (scatter)")
                                    st.line_chart(pd.DataFrame({
                                        "Actual": out_df["Compressive_Strength_MPa"],
                                        "Predicted": out_df["Predicted_Compressive_Strength_MPa"]
                                    }))
                                else:
                                    st.markdown("#### Predicted distribution")
                                    st.bar_chart(out_df["Predicted_Compressive_Strength_MPa"].reset_index(drop=True))
                                # log event
                                log_event("PREDICT_BATCH", f"rows={len(out_df)}, file={uploaded.name}")
                            except Exception as e:
                                st.error("Prediction failed: " + str(e))
                                log_event("ERROR", f"Batch predict failed: {e}")
    else:
        st.info("Upload a file to perform batch predictions or use single-sample inputs.")

st.markdown("---")
st.header("App & Model Status")
if MODEL is None:
    st.error("Model not loaded. Provide model files (local) or set MODEL_URL/SCALERX_URL/SCALERY_URL environment variables.")
else:
    st.success("Model loaded.")
    try:
        st.write("Model type:", type(MODEL).__name__)
        if hasattr(MODEL, "n_iter_"):
            st.write("Training iterations (n_iter_):", MODEL.n_iter_)
        if hasattr(MODEL, "loss_"):
            st.write("Final training loss (loss_):", round(MODEL.loss_, 6))
    except Exception:
        pass

st.markdown("### Notes & Disclaimer")
st.write("- Predictions are approximate and for research/demo only. Validate with experiments before design use.")
st.write("- If deploying publicly, consider adding authentication, stricter rate-limits, and move model files to secure storage.")
