import json
import sys
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
import sklearn.compose._column_transformer as _ct
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import src.features  # Wajib di-import sebelum joblib.load()


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

FOLDER_ARTEFAK = ROOT_DIR / "artifacts"
if not FOLDER_ARTEFAK.exists():
    FOLDER_ARTEFAK = ROOT_DIR / "models"

model = joblib.load(FOLDER_ARTEFAK / "model.joblib")

preprocess = model.named_steps.get("preprocess")
if preprocess is not None and hasattr(preprocess, "transformers_"):
    for _, trans, _ in preprocess.transformers_:
        if hasattr(trans, "named_steps"):
            for step in trans.named_steps.values():
                if not hasattr(step, "_fill_dtype") and hasattr(step, "_fit_dtype"):
                    step._fill_dtype = step._fit_dtype
with open(FOLDER_ARTEFAK / "metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

THRESHOLD = float(metadata["threshold"])

app = FastAPI(
    title="API Prediksi Cacat Injection Molding",
    description="Layanan prediksi kualitas produk injection molding berbasis XGBoost (Tim KenjiYasmin).",
    version=metadata["versi_model"],
)


class InputMolding(BaseModel):
    mesin: Literal["IM-01", "IM-02", "IM-03", "IM-04"] = Field(..., example="IM-01")
    shift: Literal["pagi", "siang", "malam"] = Field(..., example="pagi")
    bahan: str = Field(..., example="PP")
    suhu_leleh_C: float = Field(..., gt=0, example=240.0)
    suhu_cetakan_C: float = Field(..., gt=0, example=55.0)
    tekanan_injeksi_bar: float = Field(..., gt=0, example=900.0)
    kecepatan_injeksi_mm_s: float = Field(..., gt=0, example=80.0)
    waktu_pendinginan_s: float = Field(..., gt=0, example=18.0)
    waktu_siklus_s: float = Field(..., gt=0, example=32.0)
    kelembaban_bahan_pct: float = Field(..., ge=0, example=0.03)


@app.get("/")
def root():
    return {
        "status": "aktif",
        "model": metadata["algoritma"],
        "versi_model": metadata["versi_model"],
        "threshold": THRESHOLD,
    }


@app.post("/predict")
def predict(data: InputMolding):
    try:
        row = data.model_dump()
        row["mesin"] = row["mesin"].strip().upper()
        row["shift"] = row["shift"].strip().lower()
        row["bahan"] = row["bahan"].strip().upper()

        df = pd.DataFrame([row])
        prob = float(model.predict_proba(df)[:, 1][0])
        prediksi = int(prob >= THRESHOLD)
        return {
            "probabilitas_cacat": round(prob, 4),
            "threshold": THRESHOLD,
            "prediksi_cacat": prediksi,
            "status_qc": "BERISIKO CACAT" if prediksi == 1 else "NORMAL",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))