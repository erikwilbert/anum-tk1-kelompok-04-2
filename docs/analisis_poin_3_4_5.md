# Analisis Teknis dan Teoretis Soal 2 (Poin iii, iv, v)
**Penanggung Jawab: Person A (Erik Wilbert — NPM: 2406495376)**
**Kelompok: 04 (Genap) — Analisis Numerik Gasal 2026/2027**

---

## 1. Poin iii: Penyelesaian via Persamaan Normal

### A. Formulasi Matematis
Persoalan kuadrat terkecil linier (Least Squares Problem / LSP) didefinisikan sebagai:
$$\min_{\vec{x}} \| A\vec{x} - \vec{b} \|_2^2$$

Turunan pertama terhadap $\vec{x}$ menghasilkan sistem **Persamaan Normal**:
$$A^T A \vec{x} = A^T \vec{b}$$

Di mana untuk matriks $A \in \mathbb{R}^{M \times N}$ dengan rank penuh ($rank(A) = N = 6$ dan $M \gg N$):
* $A^T A$ adalah matriks simetris berukuran $6 \times 6$ dan definit positif (SPD).
* $A^T \vec{b}$ adalah vektor berukuran $6 \times 1$.

Sistem diselesaikan secara manual menggunakan **Eliminasi Gauss dengan Partial Pivoting (GEPP)** untuk menjaga kestabilan pivot diagonal, diikuti dengan **Substitusi Mundur (Back Substitution)**.

### B. Analisis Condition Number dan Stabilitas Numerik
Nilai *condition number* berbasis norma-2 didefinisikan sebagai rasio nilai singular terbesar terhadap terkecil:
$$\kappa_2(A) = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}$$

Untuk matriks normal $A^T A$:
$$\kappa_2(A^T A) = \frac{\sigma_{\max}(A^T A)}{\sigma_{\min}(A^T A)} = \frac{(\sigma_{\max}(A))^2}{(\sigma_{\min}(A))^2} = \left(\kappa_2(A)\right)^2$$

**Implikasi Presisi dan Kehilangan Informasi (*Loss of Precision*):**
1. Dalam aritmatika titik kambang IEEE 754 *double precision* (presisi $\approx 10^{-16}$, sekitar 15–16 digit signifikan), batas toleransi kesalahan galat pembulatan adalah $O(\epsilon_{\text{mach}} \cdot \kappa_2)$.
2. Jika matriks desain $A$ memiliki $\kappa_2(A) \approx 10^4$, maka $\kappa_2(A^T A) \approx 10^8$. Artinya, sekitar 8 digit presisi hilang akibat pembentukan $A^T A$.
3. Jika kondisi matriks $A$ semakin buruk (misal $\kappa_2(A) \ge 10^8$), maka $\kappa_2(A^T A) \ge 10^{16}$, yang menyebabkan matriks $A^T A$ menjadi singular secara komputasi (*numerically singular*), sehingga solusi $\vec{x}$ menjadi tidak bermakna atau mengandung *catastrophic cancellation*.

---

## 2. Poin iv: Dekomposisi QR Berbasis Givens Rotations (Kelompok Genap)

### A. Justifikasi Teoretis Pemilihan Givens Rotations
Berdasarkan penugasan kelompok genap, digunakan **Givens Rotations** dengan justifikasi teknis:
1. **Kestabilan Ortogonal Unconditional**: Matriks transformasi ortogonal $Q$ mempertahankan norma-2 Euclidean ($\|Q\vec{v}\|_2 = \|\vec{v}\|_2$). Kondisi sistem segitiga atas $R$ mempertahankan kondisi matriks awal:
   $$\kappa_2(R) = \kappa_2(A)$$
   Tidak ada pengkuadratan *condition number* seperti pada Persamaan Normal.
2. **Kesesuaian dengan Struktur Matriks SETAR**:
   Matriks desain $A$ memiliki struktur blok dengan banyak elemen nol akibat fungsi indikator rezim $I_t$:
   $$I_t = \mathbb{I}(R_{t-1} \ge 0)$$
   Pada baris Bullish ($I_t = 1$), kolom 3, 4, 5 bernilai nol. Sebaliknya pada baris Bearish ($I_t = 0$), kolom 0, 1, 2 bernilai nol.
   *Givens Rotations* bekerja secara selektif elemen-per-elemen. Jika suatu elemen sub-diagonal sudah bernilai 0 ($A_{i,j} = 0$), rotasi dapat diabaikan (*skip*) tanpa membuang komputasi, menjadikannya jauh lebih efisien dibanding *Householder reflection* yang harus memproses satu blok vektor penuh.
3. **Efisiensi Memori (In-Place & Matriks Implisit)**:
   Kita tidak perlu membentuk matriks $Q$ berukuran $M \times M$ secara eksplisit. Rotasi diaplikasikan langsung pada baris matriks $A$ dan vektor target $\vec{b}$ secara *in-place*.

### B. Formulasi Parameter Givens Stabil
Untuk mengeliminasi elemen sub-diagonal $b = A_{i, j}$ menggunakan pivot $a = A_{j, j}$, dicari $c = \cos\theta$ dan $s = \sin\theta$ yang memenuhi:
$$\begin{bmatrix} c & -s \\ s & c \end{bmatrix} \begin{bmatrix} a \\ b \end{bmatrix} = \begin{bmatrix} r \\ 0 \end{bmatrix}$$

Algoritma stabil (menghindari *overflow/underflow* kuadrat):
* Jika $|b| > |a|$: $\tau = -a/b$, $s = \frac{1}{\sqrt{1 + \tau^2}}$, $c = s \cdot \tau$
* Jika $|a| \ge |b|$: $\tau = -b/a$, $c = \frac{1}{\sqrt{1 + \tau^2}}$, $s = c \cdot \tau$

### C. Verifikasi Langkah Awal Eliminasi ($G_1 A$)
Elemen sub-diagonal pertama di kolom ke-0 adalah baris $i=1$ ($A_{1,0}$), dengan pivot $a = A_{0,0}$.
Matriks rotasi $G_1 \in \mathbb{R}^{M \times M}$ adalah:
$$G_1 = \begin{bmatrix}
c & -s & 0 & \dots & 0 \\
s & c & 0 & \dots & 0 \\
0 & 0 & 1 & \dots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \dots & 1
\end{bmatrix}$$

Pada matriks ilustrasi Halaman 10 PDF:
$A_{0,0} = 0$, $A_{1,0} = 1 \implies c = 0$, $s = 1$.
Hasil perkalian $G_1 A$ menukar baris 0 dan baris 1 serta menghasilkan baris ke-1 kolom ke-0 bernilai $0$ secara eksak.

---

## 3. Poin v: Analisis Komparatif Performa dan Pengaruh Outlier

### A. Tabel Perbandingan Teoretis (Simulasi $M=300, N=6$)

| Parameter Komparasi | Persamaan Normal ($A^T A \vec{x} = A^T\vec{b}$) | Givens Rotations QR ($R\vec{x} = Q^T\vec{b}$) |
|---|---|---|
| **Kompleksitas FLOPs Teoretis** | $\approx M N^2 + 2MN + \frac{2}{3}N^3 \approx \mathbf{14.580\text{ FLOPs}}$ | $\approx 3MN^2 - N^3 \approx \mathbf{32.220\text{ FLOPs}}$ |
| **Kebutuhan Memori Tambahan** | $O(N^2) = 336\text{ bytes}$ (Matriks $A^TA$ dan vektor $A^Tb$) | $O(1) = 48\text{ bytes}$ (In-place QR, rotasi simultan ke $\vec{b}$) |
| **Memori jika $Q$ Eksplisit** | Tidak memerlukan $Q$ | $O(M^2) = 720.000\text{ bytes}$ (Jika simpan $Q \in \mathbb{R}^{300 \times 300}$) |
| **Kestabilan Numerik** | Rentan (*ill-conditioned* karena $\kappa_2 \to \kappa_2^2$) | Sangat stabil (*unconditionally stable*, $\kappa_2(R) = \kappa_2(A)$) |
| **Norm Residual $\|A\vec{x}_{LS} - \vec{b}\|_2$** | Identik pada kondisi well-conditioned, degradasi pada ill-conditioned | Konsisten meminimalkan residual secara optimal |

### B. Analisis Dampak Outlier Ekstrem (*Market Shock / Flash Crash*)
Pada deret waktu finansial, guncangan pasar dapat menghasilkan return ekstrem (misal $-15\%$ atau $+20\%$).
1. **Fungsi Objektif Kuadrat Terkecil**:
   $$J(\vec{x}) = \sum_{t=1}^M \left( R_t - \hat{R}_t \right)^2$$
2. **Efek Kuadratik (*Square Penalty*)**:
   Karena error dikuadratkan, sebuah observasi *flash crash* dengan residual $e_t = 0.15$ menyumbang galat $e_t^2 = 0.0225$. Nilai ini ratusan kali lipat lebih besar dibanding residual data normal ($0.01^2 = 0.0001$).
3. **Konsekuensi Estimasi**:
   Sistem kuadrat terkecil akan membelokkan estimasi seluruh koefisien model $\vec{x}$ secara drastis demi memperkecil residual pada satu titik outlier tersebut. Hal ini merusak generalisasi model pada rezim normal.
4. **Rekomendasi Solusi**:
   Untuk data deret waktu riil dengan volatilitas ekstrem, pendekatan yang lebih kokoh (*robust regression*) adalah menggunakan **Huber Loss** atau norm-$L_1$ (Least Absolute Deviations / LAD).

---

## 4. Pseudocode Algoritma

### Algoritma 1: Solver Persamaan Normal via GEPP
```text
Algoritma: Solver_Persamaan_Normal(A, b)
Input : Matriks desain A berukuran M x N, vektor target b berukuran M
Output: Vektor solusi x berukuran N

1. Inisialisasi:
     ATA <- Matriks nol berukuran N x N
     ATb <- Vektor nol berukuran N
2. Hitung ATA = A^T * A dan ATb = A^T * b:
     Untuk i dari 0 hingga N - 1:
       Untuk j dari i hingga N - 1:
         ATA[i, j] <- Dot_Product(A[:, i], A[:, j])
         ATA[j, i] <- ATA[i, j]
       ATb[i] <- Dot_Product(A[:, i], b)

3. Eliminasi Gauss dengan Partial Pivoting pada [ATA | ATb]:
     Untuk k dari 0 hingga N - 2:
       Cari baris pivot p >= k dengan |ATA[p, k]| maksimum
       Tukar baris k dan baris p pada ATA dan ATb
       Untuk i dari k + 1 hingga N - 1:
         faktor <- ATA[i, k] / ATA[k, k]
         ATA[i, k:] <- ATA[i, k:] - faktor * ATA[k, k:]
         ATb[i]     <- ATb[i] - faktor * ATb[k]

4. Substitusi Mundur (Back Substitution):
     x <- Vektor nol berukuran N
     Untuk i dari N - 1 turun ke 0:
       x[i] <- (ATb[i] - Dot_Product(ATA[i, i+1:], x[i+1:])) / ATA[i, i]

5. Return x
```

### Algoritma 2: Solver Faktorisasi QR via Givens Rotations
```text
Algoritma: Solver_Givens_QR(A, b)
Input : Matriks desain A berukuran M x N, vektor target b berukuran M
Output: Vektor solusi x_LS berukuran N, matriks R berukuran N x N

1. Salin R <- Salin(A), d <- Salin(b)

2. Proses Eliminasi Rotasi Givens:
     Untuk j dari 0 hingga N - 1:
       Untuk i dari j + 1 hingga M - 1:
         Jika |R[i, j]| > 1e-15 maka:
           a <- R[j, j], b_val <- R[i, j]
           Jika |b_val| > |a| maka:
             tau <- -a / b_val
             s <- 1.0 / Sqrt(1.0 + tau^2)
             c <- s * tau
           Lainnya:
             tau <- -b_val / a
             c <- 1.0 / Sqrt(1.0 + tau^2)
             s <- c * tau

           baris_j <- R[j, j:], baris_i <- R[i, j:]
           R[j, j:] <- c * baris_j - s * baris_i
           R[i, j:] <- s * baris_j + c * baris_i
           R[i, j]  <- 0.0

           dj <- d[j], di <- d[i]
           d[j] <- c * dj - s * di
           d[i] <- s * dj + c * di

3. Ambil submatriks segitiga atas R_top <- R[0:N, 0:N] dan d_top <- d[0:N]

4. Substitusi Mundur:
     x_LS <- Vektor nol berukuran N
     Untuk i dari N - 1 turun ke 0:
       x_LS[i] <- (d_top[i] - Dot_Product(R_top[i, i+1:], x_LS[i+1:])) / R_top[i, i]

5. Return x_LS, R_top
```
