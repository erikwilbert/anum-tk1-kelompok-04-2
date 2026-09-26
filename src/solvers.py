"""
Modul Solver Numerik untuk Soal 2: Model SETAR
Penanggung Jawab: Person A (Erik Wilbert - Numerical & Algorithm Engineer)
Metode Kelompok Genap: Givens Rotations & Persamaan Normal
Sub-Poin: iii, iv, v + Pseudocode
"""

import time
import numpy as np


# =============================================================================
# 1. FUNGSI DASAR: SUBSTITUSI MUNDUR & ROTASI GIVENS
# =============================================================================

def back_substitution(U: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Menyelesaikan sistem persamaan linear segitiga atas: U x = y
    menggunakan algoritma substitusi mundur (Back Substitution) dari nol.

    Parameters:
    -----------
    U : np.ndarray, shape=(n, n)
        Matriks segitiga atas non-singular.
    y : np.ndarray, shape=(n,)
        Vektor ruas kanan.

    Returns:
    --------
    x : np.ndarray, shape=(n,)
        Solusi sistem persamaan linear.
    """
    n = len(y)
    x = np.zeros(n, dtype=np.float64)

    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) < 1e-15:
            raise ValueError(f"Pivot mendekati nol pada baris {i}: U[{i},{i}] = {U[i, i]:.2e}")
        # x_i = (y_i - sum_{j=i+1}^{n-1} U_{i,j} * x_j) / U_{i,i}
        dot_product = np.dot(U[i, i + 1:], x[i + 1:])
        x[i] = (y[i] - dot_product) / U[i, i]

    return x


def givens_rotation_params(a: float, b: float) -> tuple[float, float]:
    """
    Menghitung parameter rotasi Givens c = cos(theta) dan s = sin(theta)
    yang stabil secara numerik (menghindari overflow/underflow):
        [ c  -s ] [ a ]   [ r ]
        [ s   c ] [ b ] = [ 0 ]

    Parameters:
    -----------
    a : float
        Elemen diagonal/pivot (A_jj).
    b : float
        Elemen subdiagonal yang ingin dieliminasi menjadi 0 (A_ij).

    Returns:
    --------
    c, s : tuple[float, float]
        Nilai cosinus dan sinus dari rotasi Givens.
    """
    if abs(b) < 1e-15:
        return 1.0, 0.0

    if abs(b) > abs(a):
        tau = -a / b
        s = 1.0 / np.sqrt(1.0 + tau * tau)
        c = s * tau
    else:
        tau = -b / a
        c = 1.0 / np.sqrt(1.0 + tau * tau)
        s = c * tau

    return float(c), float(s)


# =============================================================================
# 2. POIN iii: PENYELESAIAN VIA PERSAMAAN NORMAL (A^T A x = A^T b)
# =============================================================================

def gaussian_elimination_solve(M: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Menyelesaikan SPL M x = v menggunakan Eliminasi Gauss dengan Partial Pivoting
    (GEPP) manual dari nol.
    """
    A_aug = M.astype(np.float64, copy=True)
    b_aug = v.astype(np.float64, copy=True)
    n = len(b_aug)

    for k in range(n - 1):
        # 1. Partial Pivoting: cari baris dengan elemen absolut terbesar di kolom k
        pivot_idx = k + np.argmax(np.abs(A_aug[k:, k]))
        if abs(A_aug[pivot_idx, k]) < 1e-15:
            raise ValueError(f"Matriks singular/rank-deficient pada kolom {k}.")

        # Swap baris jika pivot bukan pada baris k
        if pivot_idx != k:
            A_aug[[k, pivot_idx]] = A_aug[[pivot_idx, k]]
            b_aug[[k, pivot_idx]] = b_aug[[pivot_idx, k]]

        # 2. Eliminasi baris di bawah baris pivot
        pivot_val = A_aug[k, k]
        for i in range(k + 1, n):
            factor = A_aug[i, k] / pivot_val
            A_aug[i, k:] -= factor * A_aug[k, k:]
            A_aug[i, k] = 0.0
            b_aug[i] -= factor * b_aug[k]

    return back_substitution(A_aug, b_aug)


def solve_normal_equations(A: np.ndarray, b: np.ndarray) -> dict:
    """
    Menyelesaikan Least Squares Problem via Persamaan Normal: A^T A x = A^T b
    menggunakan pembentukan normal equations + eliminasi Gauss manual.

    Parameters:
    -----------
    A : np.ndarray, shape=(M, N)
    b : np.ndarray, shape=(M,)

    Returns:
    --------
    dict:
        'x': np.ndarray (shape=(N,)), solusi parameter x_Normal
        'residual_norm': float, ||A x - b||_2
        'cond_A': float, condition number kappa_2(A)
        'cond_ATA': float, condition number kappa_2(A^T A)
        'execution_time_ms': float, waktu eksekusi dalam milidetik
    """
    start_time = time.perf_counter()

    # 1. Bentuk matriks normal M = A^T A dan vektor ruas kanan v = A^T b
    ATA = np.dot(A.T, A)
    ATb = np.dot(A.T, b)

    # 2. Selesaikan ATA x = ATb dengan eliminasi Gauss manual
    x = gaussian_elimination_solve(ATA, ATb)

    exec_time = (time.perf_counter() - start_time) * 1000.0

    # 3. Hitung norm residual ||A x - b||_2
    res = np.dot(A, x) - b
    res_norm = float(np.linalg.norm(res, ord=2))

    # 4. Hitung condition number kappa_2(A) dan kappa_2(A^T A) menggunakan nilai singular
    s_A = np.linalg.svd(A, compute_uv=False)
    cond_A = float(s_A[0] / s_A[-1]) if s_A[-1] > 1e-15 else np.inf

    s_ATA = np.linalg.svd(ATA, compute_uv=False)
    cond_ATA = float(s_ATA[0] / s_ATA[-1]) if s_ATA[-1] > 1e-15 else np.inf

    return {
        "x": x,
        "residual_norm": res_norm,
        "cond_A": cond_A,
        "cond_ATA": cond_ATA,
        "execution_time_ms": exec_time
    }


# =============================================================================
# 3. POIN iv: PENYELESAIAN VIA FAKTORISASI QR (GIVENS ROTATIONS)
# =============================================================================

def get_first_givens_elimination(A: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Verifikasi Langkah Eliminasi Awal (Poin iv - Kelompok Genap):
    Memformulasikan representasi matriks rotasi G_1 yang mengeliminasi elemen
    sub-diagonal pertama matriks A (yaitu A[1, 0] menggunakan pivot A[0, 0]),
    lalu memverifikasi hasilnya dengan menghitung G_1 A.

    Returns:
    --------
    c, s : float, float
        Parameter rotasi cosinus dan sinus.
    G1 : np.ndarray, shape=(M, M)
        Matriks rotasi Givens eksplisit ukuran M x M.
    G1_A : np.ndarray, shape=(M, N)
        Hasil perkalian matriks G1 @ A di mana baris ke-1 kolom ke-0 bernilai 0.
    """
    M, N = A.shape
    c, s = givens_rotation_params(A[0, 0], A[1, 0])

    # Bangun matriks G1 ukuran M x M (Identitas dengan blok [0:2, 0:2] diganti rotasi)
    G1 = np.eye(M, dtype=np.float64)
    G1[0, 0] = c
    G1[0, 1] = -s
    G1[1, 0] = s
    G1[1, 1] = c

    G1_A = np.dot(G1, A)
    return c, s, G1, G1_A


def solve_givens_qr(A: np.ndarray, b: np.ndarray) -> dict:
    """
    Menyelesaikan Least Squares Problem via Faktorisasi QR berbasis Givens Rotations
    secara in-place tanpa membentuk matriks Q ukuran besar (sangat hemat memori),
    kemudian dilanjutkan dengan Back Substitution.

    Parameters:
    -----------
    A : np.ndarray, shape=(M, N) dengan M >= N
    b : np.ndarray, shape=(M,)

    Returns:
    --------
    dict:
        'x': np.ndarray (shape=(N,)), solusi kuadrat terkecil x_LS
        'residual_norm': float, ||A x_LS - b||_2
        'R': np.ndarray (shape=(N, N)), matriks segitiga atas tereduksi
        'c_first': float, cosinus rotasi pertama
        's_first': float, sinus rotasi pertama
        'execution_time_ms': float, waktu eksekusi dalam milidetik
    """
    start_time = time.perf_counter()

    R = A.astype(np.float64, copy=True)
    d = b.astype(np.float64, copy=True)
    M, N = R.shape

    c_first, s_first = 0.0, 0.0
    first_step_recorded = False

    # Proses eliminasi Givens: kolom demi kolom (j), baris demi baris (i)
    for j in range(N):
        for i in range(j + 1, M):
            if abs(R[i, j]) > 1e-15:
                c, s = givens_rotation_params(R[j, j], R[i, j])

                if not first_step_recorded and j == 0 and i == 1:
                    c_first, s_first = c, s
                    first_step_recorded = True

                # Aplikasikan rotasi Givens pada baris j dan baris i dari R
                row_j = R[j, j:].copy()
                row_i = R[i, j:].copy()
                R[j, j:] = c * row_j - s * row_i
                R[i, j:] = s * row_j + c * row_i
                R[i, j] = 0.0  # pastikan nol eksak

                # Aplikasikan rotasi Givens pada vektor ruas kanan d = Q^T b
                dj = d[j]
                di = d[i]
                d[j] = c * dj - s * di
                d[i] = s * dj + c * di

    # Ambil submatriks segitiga atas R (N x N) dan vektor d_top (N,)
    R_top = R[:N, :N]
    d_top = d[:N]

    # Selesaikan R_top * x = d_top dengan Back Substitution
    x_LS = back_substitution(R_top, d_top)

    exec_time = (time.perf_counter() - start_time) * 1000.0

    # Residual norm dapat diambil langsung dari sisa vektor d[N:] atau ||A x_LS - b||
    res = np.dot(A, x_LS) - b
    res_norm = float(np.linalg.norm(res, ord=2))

    return {
        "x": x_LS,
        "residual_norm": res_norm,
        "R": R_top,
        "c_first": c_first,
        "s_first": s_first,
        "execution_time_ms": exec_time
    }


# =============================================================================
# 4. POIN v: ANALISIS KOMPARATIF PERFORMA & FLOPs TEORETIS
# =============================================================================

def compute_theoretical_flops(M: int, N: int) -> dict:
    """
    Menghitung jumlah operasi titik kambang (FLOPs) teoretis
    antara Persamaan Normal dan Givens Rotations.

    Persamaan Normal:
    - Pembentukan A^T A (simetris): M * N^2 FLOPs
    - Pembentukan A^T b: 2 * M * N FLOPs
    - Eliminasi Gauss pada N x N: (2/3) * N^3 FLOPs
    - Back substitution: N^2 FLOPs
    Total Normal ~ M * N^2 + 2 * M * N + (2/3) * N^3 + N^2

    Givens Rotations (In-place QR):
    - Untuk setiap elemen subdiagonal yang dieliminasi (baris i, kolom j):
      * Hitung c, s: ~6 FLOPs
      * Rotasi 2 baris (panjang N - j): 4 * (N - j) FLOPs
      * Rotasi ruas kanan b: 4 FLOPs
    Total Givens ~ 3 * M * N^2 - N^3 + N^2
    """
    normal_flops = int(M * (N ** 2) + 2 * M * N + (2 / 3) * (N ** 3) + (N ** 2))
    givens_flops = int(3 * M * (N ** 2) - (N ** 3) + (N ** 2))

    # Kebutuhan memori tambahan (dalam satuan elemen float64):
    # Persamaan normal: butuh simpan ATA (N x N) dan ATb (N)
    normal_mem_elements = N * N + N
    # Givens implisit: in-place modifikasi A (M x N) dan b (M), memori tambahan O(1)
    # Jika Givens eksplisit (simpan Q): M x M elemen
    givens_mem_implicit = N  # hanya butuh buffer baris kecil
    givens_mem_explicit = M * M

    return {
        "M": M,
        "N": N,
        "normal_flops": normal_flops,
        "givens_flops": givens_flops,
        "normal_mem_bytes": normal_mem_elements * 8,
        "givens_mem_implicit_bytes": givens_mem_implicit * 8,
        "givens_mem_explicit_bytes": givens_mem_explicit * 8,
    }
