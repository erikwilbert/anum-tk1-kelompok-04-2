# Tugas Kelompok 1 — Analisis Numerik (Gasal 2026/2027)
**Sistem Persamaan Linear dan Least Squares Problem**

Repositori ini memuat penyelesaian **Soal Nomor 2**: deteksi rezim pasar dan prediksi return saham dengan model SETAR 2-rezim ($p = 2$, threshold $c = 0$), diformulasikan sebagai Least Squares Problem. Metode kelompok genap: **Givens Rotations** (QR) dan **Persamaan Normal**, keduanya diimplementasikan dari nol tanpa pustaka penyelesaian SPL/LSP.

---

## Anggota Kelompok 04 (Genap)
| No | Nama | NPM | Peran & Tanggung Jawab |
|---|---|---|---|
| 1 | **Erik Wilbert** | `2406495376` | Soal Nomor 2: Poin iii, iv, v, dan Pseudocode |
| 2 | *[Nama Anggota 2]* | *[NPM]* | Soal Nomor 2: Poin i, ii, vi, vii, dan README.md |
| 3 | *[Nama Anggota 3]* | *[NPM]* | Soal Nomor 1: Analisis Perpindahan Penumpang Antarhalte |
| 4 | *[Nama Anggota 4]* | *[NPM]* | Soal Nomor 1: Analisis Perpindahan Penumpang Antarhalte |

---

## Struktur Direktori
```text
.
├── .gitignore
├── requirements.txt
├── README.md
├── Kontrak_Kerja_Nomor_2_SETAR.pdf   # Dokumen spesifikasi kontrak antarmuka
├── TK_1_Anum_Gasal_26_27.pdf         # Soal tugas kelompok
├── generate_contract_pdf.py          # Pembuat PDF kontrak kerja (membutuhkan reportlab)
├── main.py                           # Entry point: pipeline lengkap Soal 2
├── test_solvers.py                   # Uji mandiri solver (matriks toy Hal. 10 PDF)
├── investigasi_poin_2.py             # Angka-angka investigasi numerik matriks A (poin ii)
├── evaluasi_poin_6_7.py              # Evaluasi out-of-sample & interpretasi koefisien (poin vi, vii)
├── data/
│   ├── .gitkeep
│   ├── stock_train.csv               # Dataset latih (303 baris harga -> A 300x6)
│   └── stock_test.csv                # Dataset uji (103 baris harga -> A 100x6)
├── docs/
│   ├── laporan_person_b.md           # Laporan poin i, ii, vi, vii
│   └── analisis_poin_3_4_5.md        # Laporan poin iii, iv, v + pseudocode
├── outputs/                          # Dibuat otomatis oleh main.py (grafik overlay, di-.gitignore)
└── src/
    ├── __init__.py
    ├── data_pipeline.py              # Preprocessing & pembentukan matriks A, b
    ├── kondisi.py                    # Nilai singular & condition number (Jacobi satu sisi)
    ├── solvers.py                    # Solver Givens QR & Persamaan Normal
    ├── utils.py                      # Prediksi (A @ x) & RMSE
    └── visual.py                     # Grafik overlay time series
```

---

## Panduan Instalasi & Eksekusi

### 1. Prasyarat Lingkungan
Gunakan Python 3.10 atau lebih baru. Semua perintah dijalankan dari **direktori root repositori**.

```powershell
# Membuat virtual environment
python -m venv .venv

# Aktivasi virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Instalasi dependensi
pip install -r requirements.txt
```

Pada Linux/macOS, aktivasi dengan `source .venv/bin/activate`. Dependensi yang dipakai kode Soal 2: `numpy`, `pandas`, `matplotlib` (`reportlab` hanya untuk `generate_contract_pdf.py`).

### 2. Menyiapkan Data
Letakkan `stock_train.csv` dan `stock_test.csv` di folder `data/`. Berkas harus memiliki kolom `Date` dan `Close`.

### 3. Menjalankan Kode

| Perintah | Fungsi |
|---|---|
| `python test_solvers.py` | Uji mandiri kedua solver pada matriks toy Halaman 10 PDF (tidak butuh CSV) |
| `python main.py` | Pipeline lengkap: membentuk $A, b$, menjalankan kedua solver, mencetak $\kappa_2$, residual, waktu, FLOPs, RMSE Train/Test kedua metode, dan menyimpan grafik ke `outputs/` |
| `python main.py --show` | Sama seperti di atas, dan menampilkan jendela grafik |
| `python investigasi_poin_2.py` | Angka investigasi isu numerik matriks $A$ (skala kolom, $\kappa_2$, struktur blok, outlier, penskalaan kolom) |
| `python evaluasi_poin_6_7.py` | RMSE per metode dan per rezim, pembanding prediksi nol, dampak outlier uji, galat baku dan statistik $t$ koefisien, persamaan akhir model |

### 4. Keluaran
Setelah `python main.py` selesai, grafik berikut dibuat di `outputs/`:
* `overlay_train.png` — return aktual vs estimasi pada data latih.
* `overlay_train_test.png` — deret kontinu Train + Test dengan garis pembatas.

Folder `outputs/` dan berkas `*.png` masuk `.gitignore`; jalankan `python main.py` untuk menghasilkannya kembali.

### 5. Hasil yang Diharapkan (Ringkas)
| Besaran | Nilai |
|---|---|
| Dimensi $A$ (train / test) | $300 \times 6$ / $100 \times 6$ |
| $\kappa_2(A)$, $\kappa_2(A^T A)$ | $192.61$, $37096.84$ |
| RMSE Train, Persamaan Normal dan Givens QR | $8.832134 \times 10^{-3}$ |
| RMSE Test, Persamaan Normal dan Givens QR | $1.206163 \times 10^{-2}$ |

Waktu eksekusi bergantung pada mesin dan berubah antarmenjalankan; nilai lainnya deterministik.

### 6. Aturan Implementasi
Sesuai petunjuk soal, penyelesaian SPL/LSP (eliminasi Gauss, substitusi mundur, rotasi Givens) ditulis sendiri di `src/solvers.py`. Nilai singular dan *condition number* dihitung dengan `src/kondisi.py` (metode Jacobi satu sisi buatan sendiri), baik di `src/solvers.py` maupun di `investigasi_poin_2.py`, tanpa `numpy.linalg.svd`.
