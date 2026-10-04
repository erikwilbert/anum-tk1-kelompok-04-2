"""
Perhitungan nilai singular dan condition number tanpa pustaka SVD/eigen
Penanggung Jawab: Person B (Data & Financial Modeling)
Sub-Poin: ii
"""

import numpy as np


def singular_values(X: np.ndarray, max_sweeps: int = 60) -> np.ndarray:
    """
    Menghitung seluruh nilai singular X dengan metode Jacobi satu sisi (Hestenes).

    Setiap pasangan kolom (p, q) dirotasi sehingga hasil kali dalamnya menjadi nol.
    Rotasi diulang (satu putaran = satu sweep) sampai semua pasangan kolom saling
    tegak lurus. Pada saat itu, norma-2 tiap kolom adalah nilai singular X.

    Parameters:
    -----------
    X : np.ndarray, shape=(M, N)
    max_sweeps : int
        Batas jumlah sweep sebagai pengaman bila tidak konvergen.

    Returns:
    --------
    sigma : np.ndarray, shape=(N,)
        Nilai singular terurut menurun.
    """
    U = np.array(X, dtype=np.float64, copy=True)
    n = U.shape[1]
    eps = np.finfo(np.float64).eps

    for _ in range(max_sweeps):
        ada_rotasi = False
        for p in range(n - 1):
            for q in range(p + 1, n):
                alpha = np.dot(U[:, p], U[:, p])
                beta = np.dot(U[:, q], U[:, q])
                gamma = np.dot(U[:, p], U[:, q])

                if gamma == 0.0 or abs(gamma) <= eps * np.sqrt(alpha * beta):
                    continue

                zeta = (beta - alpha) / (2.0 * gamma)
                sign = 1.0 if zeta >= 0.0 else -1.0
                t = sign / (abs(zeta) + np.sqrt(1.0 + zeta * zeta))
                c = 1.0 / np.sqrt(1.0 + t * t)
                s = c * t

                col_p = U[:, p].copy()
                col_q = U[:, q].copy()
                U[:, p] = c * col_p - s * col_q
                U[:, q] = s * col_p + c * col_q
                ada_rotasi = True

        if not ada_rotasi:
            break

    sigma = np.sqrt(np.sum(U * U, axis=0))
    return np.sort(sigma)[::-1]


def cond_2(X: np.ndarray) -> float:
    """
    Menghitung condition number kappa_2(X) = sigma_max / sigma_min.
    Bernilai inf bila X tidak berpangkat penuh (sigma_min = 0).
    """
    sigma = singular_values(X)
    if sigma[-1] <= 0.0:
        return float("inf")
    return float(sigma[0] / sigma[-1])


if __name__ == "__main__":
    contoh = np.array([[3.0, 0.0], [4.0, 5.0]])
    print("Nilai singular     :", singular_values(contoh))
    print("Seharusnya (akar 45, akar 5):", np.sqrt(45.0), np.sqrt(5.0))
    print("kappa_2            :", cond_2(contoh))
