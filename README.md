# Tugas Kelompok 1 — Analisis Numerik (Gasal 2026/2027)
**Sistem Persamaan Linear dan Least Squares Problem**

Proyek ini berisi implementasi komputasi numerik untuk penyelesaian Sistem Persamaan Linear (SPL) dan Least Squares Problem (LSP) pada model **Self-Exciting Threshold Autoregressive (SETAR)** 2-rezim (Soal Nomor 2).

---

## Anggota Kelompok 04 (Genap)
| No | Nama | NPM | Peran & Tanggung Jawab |
|---|---|---|---|
| 1 | **Erik Wilbert** | `2406495376` | Soal Nomor 2: Poin iii, iv, v, dan Pseudocode |
| 2 | *[Nama Anggota 2]* | *[NPM]* | Soal Nomor 2: Poin i, ii, vi, vii, dan README.md |
| 3 | *[Nama Anggota 3]* | *[NPM]* | Soal Nomor 1: Analisis Perpindahan Penumpang Antarhalte |
| 4 | *[Nama Anggota 4]* | *[NPM]* | Soal Nomor 1: Analisis Perpindahan Penumpang Antarhalte |

---

## Metode Khusus (Kelompok Genap)
Sesuai petunjuk Soal 2 Poin iv, kelompok genap diwajibkan menggunakan:
* **Faktorisasi QR**: Metode **Givens Rotations** (manual *from scratch*, tanpa pustaka SPL/LSP).
* **Pembanding**: Metode **Persamaan Normal** ($A^T A \vec{x} = A^T \vec{b}$).
* **Verifikasi Khusus**: Menentukan representasi matriks rotasi $G_k$ yang mengeliminasi elemen sub-diagonal pertama matriks $A$, serta memverifikasi hasil $G_k A$.

---

## Struktur Direktori
```text
.
├── .gitignore
├── requirements.txt
├── README.md
├── Kontrak_Kerja_Nomor_2_SETAR.pdf   # Dokumen spesifikasi kontrak antarmuka
├── main.py                          # Entry point eksekusi pipeline lengkap
├── test_solvers.py                  # Pengujian mandiri solver numerik (Unit Test & Hal. 10 PDF)
├── data/
│   ├── .gitkeep
│   ├── stock_train.csv              # Dataset latih (303 baris observasi)
│   └── stock_test.csv               # Dataset uji (103 baris observasi)
└── src/
    ├── __init__.py
    ├── data_pipeline.py             # Preprocessing & pembentukan matriks A, b
    ├── solvers.py                   # Solver Givens QR & Persamaan Normal
    └── utils.py                     # Fungsi metrik evaluasi (RMSE) & prediksi
```

---

## Panduan Instalasi & Eksekusi

### 1. Prasyarat Lingkungan
Gunakan Python 3.10 atau lebih baru dengan virtual environment `.venv`:

```powershell
# Membuat virtual environment
python -m venv .venv

# Aktivasi virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Instalasi dependensi
pip install -r requirements.txt
```

### 2. Menjalankan Kode
* **Uji Mandiri Solver Numerik**:
  ```powershell
  python test_solvers.py
  ```
* **Menjalankan Pipeline Lengkap (Data Latih & Uji)**:
  ```powershell
  python main.py
  ```
