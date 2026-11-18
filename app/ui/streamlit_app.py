import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn.compose._column_transformer as _ct
import streamlit as st
import xgboost as xgb

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import src.features  

if not hasattr(_ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    _ct._RemainderColsList = _RemainderColsList

_orig_xgb_setstate = xgb.Booster.__setstate__


def _safe_xgb_setstate(self, state):
    try:
        _orig_xgb_setstate(self, state)
    except xgb.core.XGBoostError:
        buf = bytes(state["handle"])
        idx = buf.find(b"\x05Model{")
        if idx == -1:
            raise
        tmp = xgb.Booster()
        tmp.load_model(bytearray(buf[idx + 6 : -1]))
        self.handle = tmp.handle
        tmp.handle = None


xgb.Booster.__setstate__ = _safe_xgb_setstate

st.set_page_config(
    page_title="Prediksi Cacat Injection Molding",
    page_icon="🏭",
    layout="wide",
)


@st.cache_resource
def muat_artefak():
    folder_artefak = ROOT_DIR / "artifacts"
    if not folder_artefak.exists():
        folder_artefak = ROOT_DIR / "models"

    model = joblib.load(folder_artefak / "model.joblib")

    # Penyesuaian atribut SimpleImputer (scikit-learn 1.6.1 -> 1.7+/1.8)
    preprocess = model.named_steps.get("preprocess")
    if preprocess is not None and hasattr(preprocess, "transformers_"):
        for _, trans, _ in preprocess.transformers_:
            if hasattr(trans, "named_steps"):
                for step in trans.named_steps.values():
                    if not hasattr(step, "_fill_dtype") and hasattr(step, "_fit_dtype"):
                        step._fill_dtype = step._fit_dtype

    with open(folder_artefak / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata


def rapikan_input(df: pd.DataFrame, fitur_wajib: list[str]) -> pd.DataFrame:
    df_clean = df[fitur_wajib].copy()
    for col in ["mesin", "shift", "bahan"]:
        df_clean[col] = df_clean[col].astype(str).str.strip()
    df_clean["mesin"] = df_clean["mesin"].str.upper()
    df_clean["shift"] = df_clean["shift"].str.lower()
    df_clean["bahan"] = df_clean["bahan"].str.upper()

    kolom_numerik = [c for c in fitur_wajib if c not in ["mesin", "shift", "bahan"]]
    for col in kolom_numerik:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
    return df_clean


model, metadata = muat_artefak()
FITUR_WAJIB = metadata["fitur_input"]
THRESHOLD = float(metadata["threshold"])

st.title("Prediksi Cacat Produk Injection Molding")
st.caption("Proyek Kelompok Machine Learning · (Kenji Ahmad Darelly & Yasmin Tia Nizarini)")

# Ringkasan Performa Model
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Algoritma Terpilih", metadata["algoritma"])
col_m2.metric("PR-AUC (Test)", f"{metadata['metrik_test']['pr_auc']:.4f}")
col_m3.metric("Recall Cacat (Test)", f"{metadata['metrik_test']['recall']:.2%}")
col_m4.metric("Threshold Keputusan", f"{THRESHOLD:.2f}")

st.divider()

tab1, tab2 = st.tabs(["Prediksi Satuan (Input Manual)", "Prediksi Massal (Upload CSV)"])

# TAB 1: INPUT MANUAL
with tab1:
    st.subheader("Parameter Proses Mesin")
    with st.form("form_prediksi"):
        c1, c2, c3 = st.columns(3)
        with c1:
            mesin = st.selectbox("Mesin", ["IM-01", "IM-02", "IM-03", "IM-04"])
            shift = st.selectbox("Shift Kerja", ["pagi", "siang", "malam"])
            bahan = st.selectbox("Jenis Bahan Plastik", ["ABS", "PP", "P.P"])
        with c2:
            suhu_leleh = st.number_input("Suhu Leleh (°C)", min_value=100.0, max_value=400.0, value=240.0, step=1.0)
            suhu_cetakan = st.number_input("Suhu Cetakan (°C)", min_value=10.0, max_value=150.0, value=55.0, step=1.0)
            tekanan_injeksi = st.number_input("Tekanan Injeksi (bar)", min_value=100.0, max_value=2500.0, value=900.0, step=10.0)
            kecepatan_injeksi = st.number_input("Kecepatan Injeksi (mm/s)", min_value=5.0, max_value=300.0, value=80.0, step=1.0)
        with c3:
            waktu_pendinginan = st.number_input("Waktu Pendinginan (detik)", min_value=1.0, max_value=120.0, value=18.0, step=0.5)
            waktu_siklus = st.number_input("Waktu Siklus Total (detik)", min_value=5.0, max_value=200.0, value=32.0, step=0.5)
            kelembaban_bahan = st.number_input("Kelembaban Bahan (%)", min_value=0.0, max_value=5.0, value=0.03, step=0.01, format="%.3f")

        submit = st.form_submit_button("Prediksi Risiko Cacat")

    if submit:
        data_input = pd.DataFrame([{
            "mesin": mesin,
            "shift": shift,
            "bahan": bahan,
            "suhu_leleh_C": suhu_leleh,
            "suhu_cetakan_C": suhu_cetakan,
            "tekanan_injeksi_bar": tekanan_injeksi,
            "kecepatan_injeksi_mm_s": kecepatan_injeksi,
            "waktu_pendinginan_s": waktu_pendinginan,
            "waktu_siklus_s": waktu_siklus,
            "kelembaban_bahan_pct": kelembaban_bahan,
        }])

        df_siap = rapikan_input(data_input, FITUR_WAJIB)
        prob = float(model.predict_proba(df_siap)[:, 1][0])
        is_cacat = prob >= THRESHOLD

        st.subheader("Hasil Prediksi")
        r1, r2 = st.columns(2)
        r1.metric("Probabilitas Cacat", f"{prob:.2%}", delta=f"Threshold: {THRESHOLD:.0%}", delta_color="off")
        if is_cacat:
            r2.error(f"BERISIKO CACAT (Probabilitas >= {THRESHOLD:.2f}) — Perlu inspeksi kualitas.")
        else:
            r2.success(f"NORMAL / LOLOS QC (Probabilitas < {THRESHOLD:.2f})")

# TAB 2: UPLOAD CSV
with tab2:
    st.subheader("Upload File CSV Batch Produksi")
    st.write("Kolom wajib pada file CSV:", ", ".join(f"`{col}`" for col in FITUR_WAJIB))

    file_csv = st.file_uploader("Pilih file CSV", type=["csv"])
    if file_csv is not None:
        df_upload = pd.read_csv(file_csv)
        kolom_kurang = [c for c in FITUR_WAJIB if c not in df_upload.columns]
        if kolom_kurang:
            st.error(f"Kolom wajib berikut tidak ditemukan di CSV: {kolom_kurang}")
        else:
            df_siap = rapikan_input(df_upload, FITUR_WAJIB)
            proba_batch = model.predict_proba(df_siap)[:, 1]
            pred_batch = (proba_batch >= THRESHOLD).astype(int)

            df_hasil = df_upload.copy()
            df_hasil["probabilitas_cacat"] = np.round(proba_batch, 4)
            df_hasil["prediksi_cacat"] = pred_batch
            df_hasil["status_qc"] = np.where(pred_batch == 1, "BERISIKO CACAT", "NORMAL")

            jml_cacat = int(pred_batch.sum())
            st.info(f"Total {len(df_hasil):,} baris diproses · Terdeteksi **{jml_cacat:,}** produk berisiko cacat ({jml_cacat / len(df_hasil):.1%}).")
            st.dataframe(df_hasil.head(50))

            csv_out = df_hasil.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Hasil Prediksi (CSV)",
                data=csv_out,
                file_name="hasil_prediksi_molding.csv",
                mime="text/csv",
            )