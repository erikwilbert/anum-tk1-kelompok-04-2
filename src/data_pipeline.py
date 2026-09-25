"""
Modul Data Pipeline untuk Soal 2: Model SETAR
Penanggung Jawab: Person B (Data & Financial Modeling)
Sub-Poin: i, ii, vi, vii
"""

import numpy as np
import pandas as pd


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
    # TODO (Person B): Implementasikan pembacaan data dan pembentukan matriks A dan b
    raise NotImplementedError("Person B: Silakan implementasikan fungsi build_dataset ini.")
