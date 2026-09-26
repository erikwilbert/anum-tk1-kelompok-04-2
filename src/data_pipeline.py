"""
Modul Data Pipeline untuk Soal 2: Model SETAR
Penanggung Jawab: Person B (Data & Financial Modeling)
Sub-Poin: i, ii, vi, vii
"""

import numpy as np
import pandas as pd


def compute_returns(close_prices: np.ndarray) -> np.ndarray:
    """
    Menghitung deret return harian R_t = (P_t - P_{t-1}) / P_{t-1}.

    Parameters:
    -----------
    close_prices : np.ndarray, shape=(n_prices,)
        Harga penutupan P_1, ..., P_{n_prices}.

    Returns:
    --------
    R : np.ndarray, shape=(n_prices - 1,)
        R[k] menyimpan R_{k+1} (return pada hari ke-(k+1), untuk k = 0..n-1).
    """
    P = np.asarray(close_prices, dtype=np.float64)
    R = (P[1:] - P[:-1]) / P[:-1]
    return R


def build_dataset(csv_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Membaca CSV harga saham, menghitung return harian, dan menyusun matriks overdetermined A & target b.

    Parameters:
    -----------
    csv_path : str
        Path ke file 'stock_train.csv' atau 'stock_test.csv'.

    Returns:
    --------
    A : np.ndarray (shape=(M, 6), dtype=np.float64)
        Matriks overdetermined dengan 6 kolom terurut:
        - Kolom 0: I_t               (1 jika R_{t-1} >= 0, else 0)
        - Kolom 1: I_t * R_{t-1}     (Lag-1 Bullish)
        - Kolom 2: I_t * R_{t-2}     (Lag-2 Bullish)
        - Kolom 3: 1 - I_t           (1 jika R_{t-1} < 0, else 0)
        - Kolom 4: (1 - I_t)*R_{t-1} (Lag-1 Bearish)
        - Kolom 5: (1 - I_t)*R_{t-2} (Lag-2 Bearish)

    b : np.ndarray (shape=(M,), dtype=np.float64)
        Vektor target berupa return hari ke-t (R_t).

    actual_returns : np.ndarray (shape=(Total,), dtype=np.float64)
        Deret return harian R_t dari awal (digunakan untuk visualisasi overlay time-series).
    """
    df = pd.read_csv(csv_path)
    close_prices = df["Close"].to_numpy(dtype=np.float64)

    R = compute_returns(close_prices)
    n = len(R)

    M = n - 2
    A = np.zeros((M, 6), dtype=np.float64)
    b = np.zeros(M, dtype=np.float64)

    for idx, t in enumerate(range(3, n + 1)):
        R_t = R[t - 1]
        R_t1 = R[t - 2]
        R_t2 = R[t - 3]

        I_t = 1.0 if R_t1 >= 0.0 else 0.0

        A[idx, 0] = I_t
        A[idx, 1] = I_t * R_t1
        A[idx, 2] = I_t * R_t2
        A[idx, 3] = 1.0 - I_t
        A[idx, 4] = (1.0 - I_t) * R_t1
        A[idx, 5] = (1.0 - I_t) * R_t2

        b[idx] = R_t

    return A, b, R


if __name__ == "__main__":
    A, b, R = build_dataset("data/stock_train.csv")
    print(f"Jumlah harga (P)      : {len(R) + 1}")
    print(f"Jumlah return (R)     : {len(R)}")
    print(f"Dimensi matriks A     : {A.shape}")
    print(f"Dimensi vektor x      : (6, 1)")
    print(f"Dimensi vektor b      : {b.shape}")
    print()
    print("3 baris pertama A:")
    print(A[:3])
    print("3 target b pertama:")
    print(b[:3])
    print()
    print("Jumlah baris Bullish (I_t=1):", int(A[:, 0].sum()))
    print("Jumlah baris Bearish (I_t=0):", int(A[:, 3].sum()))