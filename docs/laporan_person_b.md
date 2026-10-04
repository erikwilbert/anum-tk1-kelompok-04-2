# Soal 2: Model SETAR — Formulasi, Isu Numerik, Evaluasi, dan Interpretasi

Kelompok 04 (Genap) · Analisis Numerik Gasal 2026/2027

Dokumen ini membahas poin i, ii, vi, dan vii. Penyelesaian dengan Persamaan Normal dan Givens Rotations, perbandingan FLOPs dan memori, serta pseudocode (poin iii–v) ada di `docs/analisis_poin_3_4_5.md`.

Semua angka di bawah berasal dari tiga program yang dapat dijalankan ulang: `python main.py`, `python investigasi_poin_2.py`, dan `python evaluasi_poin_6_7.py`. Seluruh perhitungan memakai data `stock_train.csv` untuk estimasi dan `stock_test.csv` untuk pengujian.

---

## 1. Formulasi Matriks Overdetermined (Poin i)

### 1.1 Deret return

Dari harga penutupan $P_1, \dots, P_n$ (kolom `Close`), return harian dihitung sebagai

$$R_t = \frac{P_t - P_{t-1}}{P_{t-1}}, \qquad t = 2, \dots, n.$$

Jadi $n$ harga menghasilkan $n-1$ return. Mengikuti penomoran pada soal, return pertama disebut $R_1$.

### 1.2 Sistem $A\vec{x} = \vec{b}$

Model SETAR memakai dua lag, sehingga persamaan pertama baru bisa dibentuk untuk $t = 3$ (membutuhkan $R_2$ dan $R_1$). Untuk setiap $t = 3, \dots, n-1$, dengan $I_t = \mathbb{I}(R_{t-1} \ge 0)$, satu baris matriks $A$ dan satu elemen $\vec{b}$ berbentuk

$$A_{t-2,:} = \begin{bmatrix} I_t & I_t R_{t-1} & I_t R_{t-2} & 1-I_t & (1-I_t)R_{t-1} & (1-I_t)R_{t-2} \end{bmatrix}, \qquad b_{t-2} = R_t,$$

sedangkan parameter yang dicari disusun sebagai

$$\vec{x} = \begin{bmatrix} \alpha_1 & \phi_{1,1} & \phi_{1,2} & \alpha_2 & \phi_{2,1} & \phi_{2,2} \end{bmatrix}^T.$$

Banyak baris adalah $M = (n-1) - 2 = n - 3$.

### 1.3 Dimensi

| Berkas | Jumlah harga | Jumlah return | $A$ | $\vec{x}$ | $\vec{b}$ | Baris Bullish / Bearish |
|---|---|---|---|---|---|---|
| `stock_train.csv` | 303 | 302 | $300 \times 6$ | $6 \times 1$ | $300 \times 1$ | 195 / 105 |
| `stock_test.csv` | 103 | 102 | $100 \times 6$ | $6 \times 1$ | $100 \times 1$ | 60 / 40 |

Persamaan sebanyak 300 dengan hanya 6 yang tidak diketahui, sehingga sistem ini overdetermined dan pada umumnya tidak punya solusi eksak. Karena itu $\vec{x}$ dicari sebagai solusi kuadrat terkecil, yaitu $\min_{\vec{x}} \lVert A\vec{x} - \vec{b} \rVert_2$.

### 1.4 Pengecekan dengan contoh Halaman 10

Fungsi `build_dataset` diuji pada contoh enam harga di soal, $P = [10000, 10100, 9900, 10050, 10000, 10200]^T$. Keluarannya sama dengan contoh di PDF:

$$A = \begin{bmatrix} 0 & 0 & 0 & 1 & -0.0198 & 0.0100 \\ 1 & 0.0152 & -0.0198 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 & -0.0050 & 0.0152 \end{bmatrix}, \qquad \vec{b} = \begin{bmatrix} 0.0152 \\ -0.0050 \\ 0.0200 \end{bmatrix}.$$

Baris pertama masuk rezim Bearish karena $R_2 = -0.0198 < 0$, sedangkan baris kedua masuk rezim Bullish karena $R_3 = +0.0152 \ge 0$.

### 1.5 Gambaran data

| Statistik | Train | Test |
|---|---|---|
| Periode | 2024-01-02 s.d. 2024-10-30 | 2024-10-31 s.d. 2025-02-10 |
| Rata-rata return | 0.002500 | 0.000714 |
| Simpangan baku | 0.009148 | 0.012315 |
| Minimum | -0.043570 | -0.074355 |
| Maksimum | 0.026045 | 0.027292 |
| Skewness | -0.915 | -2.727 |
| Kelebihan kurtosis | 4.22 | 13.78 |
| Proporsi return $\ge 0$ | 65.2% | 59.8% |

Kedua deret memiliki ekor yang gemuk dan miring ke kiri, artinya kerugian ekstrem lebih besar daripada keuntungan ekstrem. Data uji lebih bergejolak daripada data latih dan memuat penurunan $-7.44\%$, hampir dua kali lipat penurunan terbesar di data latih. Hal ini berpengaruh pada hasil evaluasi di bagian 3.

Satu catatan teknis: `build_dataset` dipanggil terpisah untuk tiap berkas. Return antara harga terakhir data latih dan harga pertama data uji tidak dihitung, dan dua return pertama data uji hanya dipakai sebagai lag. Sumbu waktu pada grafik gabungan Train+Test mengikuti urutan baris $A$, jadi ada selisih tiga observasi di titik batas kedua segmen.

---

## 2. Investigasi Isu Numerik pada Matriks $A$ (Poin ii)

Ada lima hal yang diperiksa pada $A$ (data latih, $300 \times 6$): skala kolom, condition number, struktur blok, korelasi antarkolom, serta outlier dan sensitivitas threshold. Setelah itu dibahas langkah mitigasinya.

### 2.1 Skala kolom

Kolom indikator (kolom 0 dan 3) hanya berisi 0 dan 1, sedangkan kolom lag berisi return harian yang besarnya sekitar $10^{-2}$.

| Kolom | Isi | Norma-2 kolom | $\max \lvert a_{ij} \rvert$ |
|---|---|---|---|
| 0 | $I_t$ | 13.9642 | 1.0000 |
| 1 | $I_t R_{t-1}$ | 0.1297 | 0.0260 |
| 2 | $I_t R_{t-2}$ | 0.1382 | 0.0436 |
| 3 | $1 - I_t$ | 10.2470 | 1.0000 |
| 4 | $(1 - I_t) R_{t-1}$ | 0.1010 | 0.0436 |
| 5 | $(1 - I_t) R_{t-2}$ | 0.0890 | 0.0420 |

Norma kolom terbesar sekitar 157 kali norma kolom terkecil. Ketimpangan ini terlihat langsung pada nilai singular $A$, yang terbelah menjadi dua kelompok:

$$\sigma(A) = \{13.9647,\ 10.2472,\ 0.1319,\ 0.0894,\ 0.0763,\ 0.0725\}.$$

Dua nilai singular besar datang dari kolom indikator, empat sisanya dari kolom lag.

### 2.2 Condition number

$$\kappa_2(A) = \frac{13.9647}{0.0725} = 192.61, \qquad \kappa_2(A^T A) = \kappa_2(A)^2 = 37096.84.$$

Nilai singular dihitung dengan metode Jacobi satu sisi yang diimplementasikan sendiri (`src/kondisi.py`): kolom-kolom $A$ dirotasi berpasangan sampai saling tegak lurus, lalu norma tiap kolom adalah nilai singularnya. Matriks $A$ berpangkat penuh (rank 6). Dengan $\epsilon_{\text{mach}} \approx 2.22 \times 10^{-16}$, batas galat relatif akibat pembulatan adalah $\epsilon_{\text{mach}} \kappa_2(A) \approx 4.3 \times 10^{-14}$ untuk metode yang bekerja langsung pada $A$ (QR), dan $\epsilon_{\text{mach}} \kappa_2(A^T A) \approx 8.2 \times 10^{-12}$ untuk Persamaan Normal. Dengan kata lain QR kehilangan sekitar dua digit, sedangkan Persamaan Normal sekitar lima digit. Keduanya masih menyisakan sekitar sebelas digit benar atau lebih, sehingga pada data ini masalah kondisi tergolong ringan. Yang tetap penting adalah alasan teoretisnya: membentuk $A^T A$ mengkuadratkan condition number.

### 2.3 Struktur blok

$I_t$ dan $1 - I_t$ tidak pernah bernilai 1 secara bersamaan, jadi setiap baris $A$ hanya terisi di tiga kolom Bullish atau tiga kolom Bearish. Tepat separuh elemen $A$ bernilai nol (195 baris Bullish, 105 baris Bearish). Akibatnya $A^T A$ berbentuk blok diagonal, dengan blok silang bernilai nol secara eksak:

$$A^T A = \begin{bmatrix} B_1 & 0 \\ 0 & B_2 \end{bmatrix}, \qquad B_1, B_2 \in \mathbb{R}^{3 \times 3}.$$

Masalah kuadrat terkecil ini dengan demikian terpecah menjadi dua regresi yang berdiri sendiri, masing-masing berukuran $195 \times 3$ dan $105 \times 3$, dengan $\kappa_2$ sebesar 182.96 (Bullish) dan 141.33 (Bearish). Ada tiga konsekuensi:

1. Rezim Bearish diestimasi dari data yang lebih sedikit sehingga koefisiennya lebih peka terhadap gangguan.
2. Bila suatu rezim punya kurang dari tiga baris, blok terkait menjadi singular dan $A$ kehilangan pangkat penuh. Pada kedua dataset hal ini tidak terjadi.
3. Banyaknya nol membuat rotasi Givens dapat dilewati untuk elemen yang sudah nol (dibahas pada poin iv).

### 2.4 Korelasi antarkolom

Di dalam satu rezim, $R_{t-1}$ selalu bertanda sama, yaitu nonnegatif pada Bullish dan negatif pada Bearish. Akibatnya kolom lag-1 cenderung sejajar dengan kolom indikator rezimnya. Kosinus sudut antarkolom:

| Pasangan kolom | Kosinus sudut |
|---|---|
| $I_t$ dan $I_t R_{t-1}$ (Bullish) | $0.798$ |
| $1-I_t$ dan $(1-I_t) R_{t-1}$ (Bearish) | $-0.681$ |
| $I_t$ dan $I_t R_{t-2}$ (Bullish) | $0.324$ |
| $1-I_t$ dan $(1-I_t) R_{t-2}$ (Bearish) | $0.132$ |

Korelasinya sedang dan belum mendekati $\pm 1$. Ini sebabnya $\kappa_2$ setelah penskalaan (bagian 2.6) tidak turun sampai 1.

### 2.5 Outlier dan sensitivitas threshold

Return latih memiliki rata-rata $0.002500$ dan simpangan baku $0.009148$. Ada tiga return dengan $\lvert z \rvert > 3$: $R_{27} = -0.043570$, $R_{209} = -0.041955$, dan $R_{264} = -0.038043$. Satu return muncul di tiga persamaan sekaligus (sebagai target, lag-1, dan lag-2), jadi satu guncangan harga memengaruhi tiga baris. $R_{27}$ merupakan elemen terbesar pada kolom 2 dan 4, dan $R_{209}$ pada kolom 5. Dampak outlier terhadap norm residual dibahas pada poin v.

Rezim ditentukan oleh tanda $R_{t-1}$. Jika return sangat dekat dengan nol, galat kecil pada harga bisa memindahkan baris ke rezim yang salah. Pada data latih tidak ada return yang tepat nol, dan nilai $\lvert R_{t-1} \rvert$ terkecil adalah $2.97 \times 10^{-6}$. Harga dicatat dengan dua desimal, sehingga galat pembulatan pada return paling besar sekitar $0.01 / P \approx 5 \times 10^{-7}$ (pada baris tersebut $P \approx 20178$). Nilai ini sekitar enam kali lebih kecil, jadi penentuan rezim pada data ini aman, walaupun margin pada baris terdekat tidak besar.

### 2.6 Langkah mitigasi

**Penskalaan kolom.** Misalkan $D = \text{diag}(\lVert a_1 \rVert_2, \dots, \lVert a_6 \rVert_2)$. Kita selesaikan $\min \lVert (A D^{-1})\vec{y} - \vec{b} \rVert_2$, lalu mengembalikan $\vec{x} = D^{-1}\vec{y}$. Hasilnya:

| Matriks | $\kappa_2$ sebelum | $\kappa_2$ sesudah |
|---|---|---|
| $A$ | 192.61 | 3.24 |
| $A^T A$ | 37096.84 | 10.48 |

Penurunan sebesar ini menunjukkan bahwa hampir seluruh condition number berasal dari perbedaan skala kolom, bukan dari korelasi antarkolom.

Langkah lain yang diterapkan atau dipertimbangkan:

- **Memakai QR sebagai solver utama**, karena bekerja langsung pada $A$ dan tidak mengkuadratkan $\kappa_2$ (poin iv).
- **Partial pivoting pada Persamaan Normal.** Diagonal $A^T A$ berkisar dari 195 sampai 0.0079, sehingga pemilihan pivot diperlukan agar eliminasi tidak membagi dengan elemen yang kecil.
- **Aritmetika `float64`** dari pembacaan harga sampai prediksi, supaya pengurangan $P_t - P_{t-1}$ tidak menghilangkan digit tanpa perlu.
- **Validasi data.** Kedua berkas CSV tidak memiliki nilai kosong, harga nonpositif, atau baris duplikat, dan kedua rezim memiliki jauh lebih dari tiga baris.
- **Outlier tetap dipertahankan**, karena merupakan bagian dari data pasar. Pengaruhnya dianalisis pada poin v.

Penskalaan kolom sendiri tidak dipasang di pipeline utama. Dengan $\kappa_2(A) = 192.61$ kehilangan presisinya kecil, sehingga cukup dengan QR, partial pivoting, dan `float64`. Penskalaan dicatat sebagai opsi bila matriks desain diperluas dengan lag yang lebih banyak.

---

## 3. Evaluasi Out-of-Sample (Poin vi)

### 3.1 RMSE

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{t=1}^{N} \left( R_t - \hat{R}_t \right)^2}, \qquad \hat{R} = A\vec{x}_{LS}.$$

| Metode | RMSE Train | RMSE Test | Test / Train |
|---|---|---|---|
| Persamaan Normal | $8.832134 \times 10^{-3}$ | $1.206163 \times 10^{-2}$ | 1.366 |
| Givens QR | $8.832134 \times 10^{-3}$ | $1.206163 \times 10^{-2}$ | 1.366 |

Kedua solver menghasilkan koefisien yang sama sampai presisi mesin: $\lVert \vec{x}_{\text{Normal}} - \vec{x}_{\text{Givens}} \rVert_2 / \lVert \vec{x}_{\text{Givens}} \rVert_2 = 9.76 \times 10^{-16}$.

### 3.2 Rekap perbandingan metode

Tabel berikut merangkum performa numerik, kondisi matriks, dan RMSE kedua metode untuk $M = 300$, $N = 6$. Perhitungan FLOPs dan memori dijelaskan pada poin v.

| Besaran | Persamaan Normal | Givens QR |
|---|---|---|
| Matriks yang dikerjakan solver | $A^T A$ ($6 \times 6$) | $R$ ($6 \times 6$) |
| $\kappa_2$ matriks tersebut | 37096.84 | 192.61 |
| $\kappa_2(A)$ | 192.61 | 192.61 |
| FLOPs teoretis | 14.580 | 32.220 |
| Memori tambahan | 336 byte | 48 byte ($Q$ tidak dibentuk) |
| Memori bila $Q$ dibentuk eksplisit | tidak diperlukan | 720.000 byte |
| $\lVert A\vec{x} - \vec{b} \rVert_2$ (train) | 0.1529770 | 0.1529770 |
| Waktu eksekusi* | sekitar 0.2 ms | sekitar 7–20 ms |
| RMSE Train | $8.832134 \times 10^{-3}$ | $8.832134 \times 10^{-3}$ |
| RMSE Test | $1.206163 \times 10^{-2}$ | $1.206163 \times 10^{-2}$ |

\*Waktu bergantung pada mesin dan beban saat dijalankan, sehingga hanya perbandingan kasarnya yang bermakna.

Pada Givens QR, $\kappa_2(R) = \kappa_2(A) = 192.61$, sesuai teori: rotasi ortogonal tidak mengubah condition number.

### 3.3 Metode mana yang lebih baik?

Dari sisi akurasi, keduanya sama: RMSE dan residualnya identik di semua digit yang ditampilkan. Dari sisi stabilitas, Givens QR lebih unggul secara teori karena $\kappa_2(R) = \kappa_2(A)$, tetapi pada data ini keunggulan itu belum terlihat. Galat teoretis Persamaan Normal (sekitar $10^{-11}$) maupun Givens (sekitar $10^{-14}$) sama-sama jauh lebih kecil daripada galat statistik model yang berorde $10^{-2}$. Dari sisi waktu, Persamaan Normal jauh lebih cepat, sejalan dengan jumlah FLOPs-nya yang lebih sedikit.

Karena itu $\vec{x}_{LS}$ dari Givens QR dipakai sebagai model terbaik, sesuai pembagian metode untuk kelompok genap. Alasannya adalah jaminan stabilitas bila kolom $A$ nanti lebih berkorelasi atau lebih timpang skalanya, bukan karena RMSE-nya lebih rendah.

### 3.4 Kemampuan generalisasi

RMSE naik dari $8.83 \times 10^{-3}$ pada data latih menjadi $1.21 \times 10^{-2}$ pada data uji. Kenaikan ini perlu dibandingkan dengan model paling sederhana, yaitu selalu menebak return $= 0$.

| Himpunan | RMSE model | RMSE tebakan nol | Perbaikan | $R^2$ |
|---|---|---|---|---|
| Train | $8.832 \times 10^{-3}$ | $9.502 \times 10^{-3}$ | 7.05% | 0.0697 |
| Test | $1.2062 \times 10^{-2}$ | $1.2424 \times 10^{-2}$ | 2.92% | 0.0532 |

Sebagian besar kenaikan RMSE berasal dari data uji yang lebih bergejolak: RMSE tebakan nol saja naik sekitar 31%, sementara RMSE model naik sekitar 37%. Dua baris target uji dengan $\lvert z \rvert > 3$ ($-7.44\%$ dan $-4.84\%$) menyumbang 54.1% dari seluruh jumlah kuadrat galat uji. Tanpa kedua baris itu, RMSE uji turun menjadi $8.26 \times 10^{-3}$. Ini sejalan dengan analisis poin v bahwa beberapa guncangan dapat mendominasi fungsi objektif kuadrat terkecil.

Model tidak mampu memprediksi datangnya guncangan. Pada hari crash $-7.44\%$ prediksinya hanya $-0.09\%$. Per rezim, model lebih baik daripada tebakan nol pada rezim Bearish di data uji ($1.432 \times 10^{-2}$ dibanding $1.528 \times 10^{-2}$), tetapi sedikit lebih buruk pada rezim Bullish ($1.029 \times 10^{-2}$ dibanding $1.008 \times 10^{-2}$).

Secara keseluruhan, model ini stabil secara numerik dan tidak ambruk di luar sampel, tetapi daya prediksinya kecil: $R^2$ pada data uji hanya sekitar 5%, dan keunggulan atas tebakan nol menyusut dari 7.05% menjadi 2.92%. Hal ini wajar untuk return harian saham. Model lebih tepat dibaca sebagai gambaran pola dua rezim daripada alat peramal.

---

## 4. Interpretasi Finansial dan Visualisasi (Poin vii)

### 4.1 Persamaan akhir

Koefisien dari Givens QR pada data latih menghasilkan

$$R_t = \begin{cases} \;\;\,0.001513 + 0.172153\, R_{t-1} + 0.166073\, R_{t-2}, & R_{t-1} \ge 0 \quad \text{(Bullish)} \\[4pt] -0.001213 - 0.360377\, R_{t-1} - 0.109808\, R_{t-2}, & R_{t-1} < 0 \quad \text{(Bearish)} \end{cases}$$

### 4.2 Koefisien dan signifikansinya

| Parameter | Estimasi | Galat baku | $t$ | Makna |
|---|---|---|---|---|
| $\alpha_1$ | $+0.001513$ | $0.001124$ | $1.35$ | Drift naik sekitar 0.15% per hari |
| $\phi_{1,1}$ | $+0.172153$ | $0.115911$ | $1.49$ | Momentum lag-1 (Bullish) |
| $\phi_{1,2}$ | $+0.166073$ | $0.069299$ | $2.40$ | Momentum lag-2 (Bullish) |
| $\alpha_2$ | $-0.001213$ | $0.001207$ | $-1.00$ | Drift turun sekitar 0.12% per hari |
| $\phi_{2,1}$ | $-0.360377$ | $0.121428$ | $-2.97$ | *Mean-reversion* lag-1 (Bearish) |
| $\phi_{2,2}$ | $-0.109808$ | $0.101792$ | $-1.08$ | *Mean-reversion* lag-2 (Bearish) |

Galat baku dihitung dari $s^2 (A^T A)^{-1}$ dengan $s^2 = \lVert A\vec{x} - \vec{b} \rVert_2^2 / (M - N)$ dan $s = 0.00892$. Matriks $(A^T A)^{-1}$ dibentuk kolom demi kolom dengan eliminasi Gauss yang sama seperti pada solver. Nilai $t$ ini mengandaikan galat model homoskedastik dan tidak berkorelasi, padahal return berekor gemuk (bagian 1.5). Karena itu nilai $t$ hanya dipakai sebagai petunjuk, bukan uji formal.

### 4.3 Rezim Bullish: cenderung melanjutkan tren

Kedua koefisien lag bernilai positif ($\phi_{1,1} = +0.172$ dan $\phi_{1,2} = +0.166$), artinya kenaikan pada satu dan dua hari sebelumnya cenderung diikuti kenaikan lagi. Contohnya, setelah kenaikan 1% (dengan $R_{t-2} = 0$) model memprediksi kenaikan $+0.32\%$. Sifat ini sesuai dengan *momentum persistence*. Namun hanya $\phi_{1,2}$ yang berjarak lebih dari dua galat baku dari nol ($t = 2.40$). Koefisien $\phi_{1,1}$ ($t = 1.49$) dan drift $\alpha_1$ ($t = 1.35$) belum cukup meyakinkan.

### 4.4 Rezim Bearish: cenderung berbalik arah

Kedua koefisien lag bernilai negatif ($\phi_{2,1} = -0.360$ dan $\phi_{2,2} = -0.110$), yang menandakan pembalikan arah (*mean-reversion*). Koefisien $\phi_{2,1}$ adalah yang terbesar dan paling signifikan di antara semuanya ($t = -2.97$): setelah penurunan 1%, prediksinya adalah $-0.1213\% + 0.3604\% = +0.24\%$, yaitu pantulan teknikal. Pantulan baru diprediksi bila penurunan kemarin lebih dari sekitar 0.34% (dengan $R_{t-2} = 0$); untuk penurunan yang lebih kecil, drift $\alpha_2$ masih mengarahkan prediksi ke bawah. Koefisien $\phi_{2,2}$ juga searah dengan pembalikan, tetapi tidak signifikan ($t = -1.08$). Tidak ada tanda *panic selling* ($\phi_{2,1} > 0$) pada data ini.

### 4.5 Perbandingan kedua rezim

Pasar yang sedang naik cenderung melanjutkan kenaikan dengan lemah, sedangkan pasar yang sedang turun cenderung memantul lebih kuat. Dari enam koefisien, hanya dua yang melewati $\lvert t \rvert = 2$, yaitu $\phi_{2,1}$ (pembalikan pada Bearish) dan $\phi_{1,2}$ (momentum pada Bullish). Dua temuan inilah yang cukup kokoh, sedangkan sisanya sebaiknya dibaca sebagai kecenderungan saja.

### 4.6 Grafik overlay

Dua grafik dibuat otomatis oleh `python main.py` dan disimpan di folder `outputs/`.

**Gambar 1.** Return aktual dan estimasi model pada data latih (`outputs/overlay_train.png`).

![Overlay return aktual vs estimasi pada data latih](../outputs/overlay_train.png)

**Gambar 2.** Deret kontinu Train+Test dengan garis pembatas dan penanda wilayah (`outputs/overlay_train_test.png`).

![Overlay return aktual vs estimasi pada data Train+Test](../outputs/overlay_train_test.png)

Beberapa hal yang terlihat pada grafik:

1. Garis estimasi jauh lebih landai daripada return aktual. Simpangan baku prediksi hanya $0.0024$ pada data latih dan $0.0035$ pada data uji, sedangkan return aktual $0.0092$ dan $0.0124$. Ini konsisten dengan $R^2$ yang kecil.
2. Lonjakan positif pada garis estimasi sesaat setelah crash di data uji adalah efek $\phi_{2,1} < 0$. Setelah $-7.44\%$, model memprediksi pantulan $+2.56\%$ (aktual $+1.16\%$), dan setelah $-4.84\%$ memprediksi $+1.49\%$ (aktual $+2.73\%$). Arahnya tepat pada kedua kasus, besarnya belum.
3. Crash itu sendiri tidak terdeteksi: prediksi pada dua hari crash masing-masing $-0.09\%$ dan $+0.13\%$. Model baru bereaksi setelah guncangan terjadi.
