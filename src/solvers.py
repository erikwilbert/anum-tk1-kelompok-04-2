"""
Modul Solver Numerik untuk Soal 2: Model SETAR
Penanggung Jawab: Person A (Erik Wilbert - Numerical & Algorithm Engineer)
Metode Kelompok Genap: Givens Rotations & Persamaan Normal
Sub-Poin: iii, iv, v + Pseudocode
"""

import numpy as np


def solve_normal_equations(A: np.ndarray, b: np.ndarray) -> dict:
    """
    Menyelesaikan Least Squares Problem via Persamaan Normal: A^T A x = A^T b
    menggunakan solver SPL manual (tanpa np.linalg.solve / np.linalg.lstsq).

    Parameters:
    -----------
    A : np.ndarray, shape=(M, 6)
    b : np.ndarray, shape=(M,)

    Returns:
    --------
    dict:
        'x': np.ndarray (shape=(6,)), solusi koefisien [alpha_1, phi_11, phi_12, alpha_2, phi_21, phi_22]
        'residual_norm': float, ||A*x - b||_2
        'cond_A': float, condition number kappa_2(A)
        'cond_ATA': float, condition number kappa_2(A^T A)
    """
    # TODO (Person A): Implementasikan solver SPL Persamaan Normal manual
    raise NotImplementedError("Person A: Implementasikan solver Persamaan Normal manual.")


def solve_givens_qr(A: np.ndarray, b: np.ndarray) -> dict:
    """
    Menyelesaikan Least Squares Problem via Dekomposisi QR berbasis Givens Rotations
    secara manual + Back Substitution.

    Parameters:
    -----------
    A : np.ndarray, shape=(M, 6)
    b : np.ndarray, shape=(M,)

    Returns:
    --------
    dict:
        'x': np.ndarray (shape=(6,)), solusi kuadrat terkecil x_LS
        'residual_norm': float, ||A*x_LS - b||_2
        'R': np.ndarray (shape=(6, 6)), matriks segitiga atas
        'G1': np.ndarray, representasi matriks rotasi pertama pengeliminasi subdiagonal
    """
    # TODO (Person A): Implementasikan faktorisasi Givens QR manual + back substitution
    raise NotImplementedError("Person A: Implementasikan Givens QR solver manual.")
