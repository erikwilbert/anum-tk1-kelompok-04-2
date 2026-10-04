# Tugas Kelompok 1 — Analisis Numerik

**Sistem Persamaan Linear dan Least Squares Problem** · Analisis Numerik A · Fakultas Ilmu Komputer, Universitas Indonesia · Gasal 2026/2027

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-only%20for%20arrays-013243?logo=numpy&logoColor=white)
![Kelompok](https://img.shields.io/badge/Kelompok-A04%20(Genap)-F7C600)

---

## Anggota Kelompok A04

| No | Nama | NPM | Soal |
|---|---|---|---|
| 1 | Erik Wilbert | 2406495376 | Nomor 2 |
| 2 | Nadia Aisyah Fazila | 2406495584 | Nomor 1 |
| 3 | Evan Haryo Widodo | 2406435824 | Nomor 1 |
| 4 | Muhammad Sabri | 2506623793 | Nomor 2 |

---

## Soal Nomor 1

Anggota: Evan Haryo Widodo dan Nadia Aisyah Fazila.

---

## Soal Nomor 2 — Prediksi Return Saham dengan Model SETAR

Manajemen *Daily Bugle* ingin memprediksi return harian saham yang berperilaku berbeda saat pasar menguat (*bullish*) dan melemah (*bearish*). Kami memakai model **SETAR 2-rezim** (*Self-Exciting Threshold Autoregressive*) dengan lag $p = 2$ dan threshold $c = 0$, lalu merumuskan estimasi parameternya sebagai **Least Squares Problem** (LSP) overdetermined.

Metode kelompok genap: **Rotasi Givens (QR)** dan **Persamaan Normal (GEPP)**. Keduanya ditulis dari nol, tanpa pustaka penyelesaian SPL/LSP.

### Pembagian Tugas Soal 2

| Poin | Pekerjaan | Penanggung jawab | Berkas |
|---|---|---|---|
| i | Formulasi matriks $A$ dan vektor $b$ dari data harga | Muhammad Sabri | `src/data_pipeline.py` |
| ii | Investigasi isu numerik matriks $A$ (skala kolom, $\kappa_2$, kolinearitas, struktur blok) dan mitigasinya | Muhammad Sabri | `investigasi_poin_2.py`, `src/kondisi.py` |
| iii | Solver Persamaan Normal (GEPP + substitusi mundur) dan analisis kehilangan presisi | Erik | `src/solvers.py` |
| iv | Faktorisasi QR Givens Rotations dan verifikasi langkah $G_1 A$ | Erik | `src/solvers.py`, `test_solvers.py` |
| v | Perbandingan FLOPs, memori, kestabilan, dan pengaruh *outlier* | Erik | `src/solvers.py`, `main.py` |
| vi | Evaluasi *out-of-sample*: prediksi dan RMSE data uji | Muhammad Sabri | `evaluasi_poin_6_7.py`, `src/utils.py` |
| vii | Interpretasi koefisien dan grafik overlay deret waktu | Muhammad Sabri | `evaluasi_poin_6_7.py`, `src/visual.py` |
| — | Pseudocode kedua algoritma | Erik | `docs/analisis_poin_3_4_5.md` |
| — | Laporan poin i, ii, vi, vii | Muhammad Sabri | `docs/laporan_person_b.md` |
| — | Laporan poin iii, iv, v | Erik | `docs/analisis_poin_3_4_5.md` |
| — | README dan integrasi pipeline | Muhammad Sabri | `README.md`, `main.py` |

### Model

Return harian dihitung dari harga penutupan:

$$R_t = \frac{P_t - P_{t-1}}{P_{t-1}}, \qquad I_t = \mathbb{1}(R_{t-1} \ge 0)$$
 
Setiap baris matriks desain $A$ dan target $b$:

$$A_{t,:} = [\,I_t,\; I_t R_{t-1},\; I_t R_{t-2},\; 1-I_t,\; (1-I_t) R_{t-1},\; (1-I_t) R_{t-2}\,], \qquad b_t = R_t$$

dengan parameter $x = [\alpha_1, \phi_{1,1}, \phi_{1,2}, \alpha_2, \phi_{2,1}, \phi_{2,2}]^T \in \mathbb{R}^6$ dan estimasi $\min_x \lVert Ax - b \rVert_2$.

| Data | Harga | Return | Ukuran $A$ | Bullish / Bearish |
|---|---|---|---|---|
| Latih (`stock_train.csv`) | 303 | 302 | $300 \times 6$ | 195 / 105 |
| Uji (`stock_test.csv`) | 103 | 102 | $100 \times 6$ | 60 / 40 |

### Dua Solver

| | Persamaan Normal | Givens QR |
|---|---|---|
| Sistem | $A^TAx = A^Tb$ | $Rx = Q^Tb$ |
| Langkah | bentuk $A^TA$, GEPP, substitusi mundur | rotasi Givens *in-place* pada $A$ dan $b$, substitusi mundur |
| $\kappa_2$ efektif | $\kappa_2(A)^2$ (dikuadratkan) | $\kappa_2(R) = \kappa_2(A)$ |
| FLOPs teoretis ($M=300$, $N=6$) | 14.580 | 32.220 |
| Memori tambahan | 336 bytes | 48 bytes (720 KB bila $Q$ disimpan eksplisit) |
| Pemanfaatan struktur | tidak ada | rotasi dilewati bila elemen sudah nol (blok nol akibat $I_t$) |

Parameter rotasi memakai formulasi Golub & Van Loan agar tidak *overflow*. Nilai singular dan *condition number* dihitung dengan metode Jacobi satu sisi buatan sendiri di `src/kondisi.py`, tanpa `numpy.linalg.svd`.

### Hasil

| Besaran | Nilai |
|---|---|
| $\kappa_2(A)$ | $192{,}61$ |
| $\kappa_2(A^TA)$ | $37.096{,}84$ ($= \kappa_2(A)^2$) |
| Norma residual $\lVert Ax - b \rVert_2$ (train) | $0{,}152977$ |
| Selisih relatif kedua solver $\lVert x_{LS} - x_{N} \rVert / \lVert x_{LS} \rVert$ | $\approx 10^{-15}$ |
| RMSE Train, kedua metode | $8{,}832134 \times 10^{-3}$ |
| RMSE Test, kedua metode | $1{,}206163 \times 10^{-2}$ |

Model akhir (koefisien Givens QR):

$$R_t = \begin{cases} 0{,}001513 + 0{,}172153\,R_{t-1} + 0{,}166073\,R_{t-2} & R_{t-1} \ge 0 \;\text{(bullish)} \\ -0{,}001213 - 0{,}360377\,R_{t-1} - 0{,}109808\,R_{t-2} & R_{t-1} < 0 \;\text{(bearish)} \end{cases}$$

Temuan utama:
- **Bullish:** $\phi_{1,1} > 0$ menandakan *momentum* (kenaikan cenderung berlanjut).
- **Bearish:** $\phi_{2,1} < 0$ menandakan *mean-reversion* (penurunan cenderung terkoreksi). Tidak ada tanda *panic selling* ($\phi_{2,1} > 0$).
- Pada data ini kedua solver sama akurat karena $\kappa_2(A)$ kecil. Givens QR dipilih karena aman bila matriks lebih buruk kondisinya (misalnya lag yang berkolinearitas tinggi).
- Penalti kuadrat L2 sensitif terhadap *outlier* ekstrem (*flash crash*). Untuk data seperti itu disarankan regresi robust (Huber atau LAD).

### Struktur Direktori

```text
.
├── main.py                    # Pipeline lengkap Soal 2
├── test_solvers.py            # Uji mandiri solver (matriks toy)
├── investigasi_poin_2.py      # Investigasi numerik matriks A (poin ii)
├── evaluasi_poin_6_7.py       # Evaluasi out-of-sample & interpretasi (poin vi, vii)
├── generate_contract_pdf.py   # Pembuat PDF kontrak kerja (butuh reportlab)
├── requirements.txt
├── data/
│   ├── stock_train.csv        # 303 baris harga -> A 300x6
│   └── stock_test.csv         # 103 baris harga -> A 100x6
├── docs/
│   ├── laporan_person_b.md    # Laporan poin i, ii, vi, vii
│   └── analisis_poin_3_4_5.md # Laporan poin iii, iv, v + pseudocode
├── outputs/                   # Grafik overlay (dibuat main.py, di-.gitignore)
└── src/
    ├── data_pipeline.py       # Return, matriks A dan vektor b
    ├── kondisi.py             # Nilai singular & condition number (Jacobi)
    ├── solvers.py             # Givens QR & Persamaan Normal
    ├── utils.py               # Prediksi (A @ x) & RMSE
    └── visual.py              # Grafik overlay time series
```

### Cara Menjalankan

Gunakan Python 3.10+ dan jalankan semua perintah dari root repositori.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Letakkan `stock_train.csv` dan `stock_test.csv` di `data/` (kolom wajib: `Date`, `Close`).

| Perintah | Fungsi |
|---|---|
| `python test_solvers.py` | Uji kedua solver pada matriks toy, termasuk verifikasi langkah $G_1 A$ (tanpa CSV) |
| `python main.py` | Pipeline lengkap: $\kappa_2$, residual, waktu, FLOPs, RMSE Train/Test, dan grafik ke `outputs/` |
| `python main.py --show` | Sama seperti di atas, dengan menampilkan jendela grafik |
| `python investigasi_poin_2.py` | Skala kolom, $\kappa_2$, struktur blok, *outlier*, penskalaan kolom |
| `python evaluasi_poin_6_7.py` | RMSE per metode dan rezim, pembanding prediksi nol, galat baku dan statistik $t$ koefisien |

Keluaran grafik: `outputs/overlay_train.png` (aktual vs estimasi pada data latih) dan `outputs/overlay_train_test.png` (deret kontinu latih + uji dengan garis pembatas).

Waktu eksekusi bergantung pada mesin. Nilai lainnya deterministik.

### Pustaka

`numpy` dipakai hanya untuk penyimpanan array dan operasi dasar, `pandas` untuk membaca CSV, `matplotlib` untuk grafik. Eliminasi Gauss, substitusi mundur, rotasi Givens, dan perhitungan nilai singular ditulis sendiri.
