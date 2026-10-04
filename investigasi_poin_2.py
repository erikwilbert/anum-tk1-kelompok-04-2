"""
Investigasi Isu Numerik Matriks Desain A (Soal 2 - Poin ii)
Penanggung Jawab: Person B (Data & Financial Modeling)
"""

import os

import numpy as np

from src.data_pipeline import build_dataset
from src.kondisi import cond_2, singular_values


def main():
    np.set_printoptions(linewidth=120, precision=4, suppress=True)

    A, _, R = build_dataset(os.path.join("data", "stock_train.csv"))
    col_norms = np.sqrt(np.sum(A * A, axis=0))
    A_scaled = A / col_norms

    print("=" * 70)
    print("INVESTIGASI ISU NUMERIK MATRIKS DESAIN A (POIN ii)")
    print("=" * 70)

    print("\n[1] Skala kolom")
    print(f"  Dimensi A            : {A.shape}")
    print(f"  Norma-2 tiap kolom   : {col_norms}")
    print(f"  max|a_ij| tiap kolom : {np.abs(A).max(axis=0)}")
    print(f"  Rasio norma max/min  : {col_norms.max() / col_norms.min():.2f}")

    print("\n[2] Kondisi matriks")
    kappa = cond_2(A)
    eps = np.finfo(np.float64).eps
    print(f"  Nilai singular A     : {singular_values(A)}")
    print(f"  kappa_2(A)           : {kappa:.4f}")
    print(f"  kappa_2(A^T A)       : {cond_2(A.T @ A):.4f}")
    print(f"  eps * kappa_2(A)     : {eps * kappa:.2e}")
    print(f"  eps * kappa_2(A^T A) : {eps * kappa ** 2:.2e}")

    print("\n[3] Struktur blok dan korelasi antarkolom")
    print(f"  Baris Bullish / Bearish : {int(A[:, 0].sum())} / {int(A[:, 3].sum())}")
    print(f"  Fraksi elemen nol       : {np.mean(A == 0.0):.2f}")
    print(f"  max|A^T A| blok silang  : {np.abs((A.T @ A)[:3, 3:]).max():.1e}")
    print("  Kosinus sudut antarkolom:")
    print(A_scaled.T @ A_scaled)

    print("\n[4] Outlier dan sensitivitas threshold")
    z = (R - R.mean()) / R.std()
    idx = np.where(np.abs(z) > 3.0)[0]
    lag1 = A[:, 1] + A[:, 4]
    print(f"  Return: mean={R.mean():.6f}, std={R.std():.6f}, min={R.min():.6f}, max={R.max():.6f}")
    print(f"  Outlier |z| > 3      : {[(f'R_{i + 1}', round(float(R[i]), 6)) for i in idx]}")
    print(f"  Return bernilai 0    : {int(np.sum(R == 0.0))}")
    print(f"  min |R_(t-1)|        : {np.abs(lag1).min():.3e}")

    print("\n[5] Mitigasi: penskalaan kolom (norma-2 tiap kolom = 1)")
    print(f"  kappa_2(A D^-1)                 : {cond_2(A_scaled):.4f}")
    print(f"  kappa_2((A D^-1)^T (A D^-1))    : {cond_2(A_scaled.T @ A_scaled):.4f}")
    print(f"  kappa_2 blok Bullish (sebelum)  : {cond_2(A[A[:, 0] == 1.0, :3]):.4f}")
    print(f"  kappa_2 blok Bearish (sebelum)  : {cond_2(A[A[:, 3] == 1.0, 3:]):.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
