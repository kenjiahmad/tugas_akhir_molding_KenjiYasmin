"""Fitur rekayasa proses injection molding. Dipakai bersama saat training dan prediksi."""
import numpy as np
import pandas as pd

def tambah_fitur(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 1. Selisih suhu leleh plastik dengan suhu cetakan (mempengaruhi laju pembekuan)
    df["delta_suhu_C"] = df["suhu_leleh_C"] - df["suhu_cetakan_C"]
    
    # 2. Rasio waktu pendinginan terhadap total waktu siklus
    df["rasio_pendinginan"] = df["waktu_pendinginan_s"] / df["waktu_siklus_s"].replace(0, np.nan)
    
    # 3. Indikator energi/laju pengisian cetakan (tekanan x kecepatan injeksi)
    df["laju_energi_injeksi"] = df["tekanan_injeksi_bar"] * df["kecepatan_injeksi_mm_s"]
    
    # 4. Indikator risiko kelembaban terhadap suhu leleh tinggi
    df["indeks_uap"] = df["kelembaban_bahan_pct"] * df["suhu_leleh_C"]
    return df
