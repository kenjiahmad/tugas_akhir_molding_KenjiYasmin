# Prediksi Cacat Produk Injection Molding · Tim KenjiYasmin

**Kasus:** 01 - Pengendalian mutu injection molding 
· **Aplikasi:** https://tugasakhirmoldingkenjiyasmin-qyqifvknuvkdqkyvkzgmza.streamlit.app/ 
· **Video:** https://drive.google.com/drive/folders/1rk7Q_rQ-CoNzu9vFC217uI51LrfgHR_X?usp=sharing 
· **Laporan:** https://drive.google.com/drive/folders/1hvBOffEWYcqv3dufo_ipnZ5w4oA0jjIL?usp=sharing

| Anggota | NIM | Bagian yang dikerjakan |
| --- | --- | --- |
| Kenji Ahmad Darelly | 2405840 | EDA, pembersihan data, *feature engineering* (`src/features.py`), pemodelan XGBoost, pengembangan REST API (FastAPI), serta *deployment* GitHub & Streamlit Cloud |
| Yasmin Tia Nizarini | 2403165 | Komparasi model & evaluasi PR-AUC, optimasi *threshold* berbasis biaya, pengembangan antarmuka web Streamlit & pengujian skenario, penyusunan laporan, serta *editing* video demo |

## Masalah dan pengguna

Pengguna aplikasi ini adalah *plant manager* dan operator *quality control* (QC) pada lini produksi *injection molding*. Model ini membantu memprediksi risiko produk cacat pada setiap siklus cetak sebelum lolos ke tahap berikutnya, sehingga inspeksi manual dapat diprioritaskan pada produk berisiko tinggi dan kerugian akibat lolosnya produk cacat (FN = Rp500.000) dapat ditekan dibandingkan biaya inspeksi ulang (FP = Rp25.000).

## Data dan pembersihan

| Masalah yang ditemukan | Jumlah baris | Cara menangani |
|---|---|---|
| Baris duplikat persis | 350 | Dihapus dengan `drop_duplicates()` sehingga tersisa 25.000 baris |
| Penulisan teks kategori tidak seragam (`shift` & `bahan`) | Seluruh kolom teks | Diseragamkan dengan `str.strip()` serta `lower()` / `upper()` |
| Nilai kosong bawaan pada `waktu_pendinginan_s` | 257 | Diisi menggunakan nilai median (`SimpleImputer`) di dalam pipeline |
| Nilai mustahil secara fisika (`tekanan_injeksi_bar <= 0`) | 125 | Diubah menjadi `NaN` lalu diimputasi median di dalam pipeline |
| Nilai mustahil secara fisika (`waktu_siklus_s <= 0`) | 75 | Diubah menjadi `NaN` lalu diimputasi median di dalam pipeline |

## Hasil

| Model | PR-AUC *validation* | PR-AUC *test* |
|---|---|---|
| *Baseline*: Dummy Acak | 0.0748 | - |
| *Baseline*: Logistic Regression | 0.4989 | - |
| Random Forest | 0.5913 | - |
| **XGBoost (Model Final)** | **0.6194** | **0.6504** |

*Threshold* yang dipakai: **0.23**, karena dengan asumsi kerugian produk cacat lolos (*False Negative*) sebesar Rp500.000 dan biaya inspeksi ulang (*False Positive*) sebesar Rp25.000, *threshold* 0.23 meminimalkan total biaya operasional dan menangkap 87,92% produk cacat pada data uji.

## Cara menjalankan

```bash
pip install -r requirements.txt
uvicorn app.api.main:app --reload       # API
streamlit run app/ui/streamlit_app.py   # aplikasi
