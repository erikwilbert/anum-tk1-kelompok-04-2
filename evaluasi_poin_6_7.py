"""
Evaluasi Out-of-Sample dan Interpretasi Finansial (Soal 2 - Poin vi & vii)
Penanggung Jawab: Person B (Data & Financial Modeling)
"""

import os

import numpy as np

from src.data_pipeline import build_dataset
from src.solvers import solve_normal_equations, solve_givens_qr, gaussian_elimination_solve
from src.utils import predict, calculate_rmse

NAMA_PARAMETER = ["alpha1", "phi1,1", "phi1,2", "alpha2", "phi2,1", "phi2,2"]


def standard_errors(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Galat baku koefisien: se_j = sqrt( s^2 * [(A^T A)^-1]_jj ),
    dengan s^2 = ||Ax - b||^2 / (M - N). Invers (A^T A)^-1 dibentuk kolom demi
    kolom memakai eliminasi Gauss buatan sendiri pada solvers.py.
    """
    M, N = A.shape
    ATA = A.T @ A
    inv_diag = np.array([gaussian_elimination_solve(ATA, np.eye(N)[:, j])[j] for j in range(N)])
    s2 = np.sum((A @ x - b) ** 2) / (M - N)
    return np.sqrt(s2 * inv_diag)


def rmse_per_rezim(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> dict:
    pred = predict(A, x)
    hasil = {}
    for nama, mask in (("Bullish", A[:, 0] == 1.0), ("Bearish", A[:, 3] == 1.0)):
        hasil[nama] = (int(mask.sum()), calculate_rmse(b[mask], pred[mask]),
                       calculate_rmse(b[mask], np.zeros(int(mask.sum()))))
    return hasil


def main():
    np.set_printoptions(linewidth=120, precision=6, suppress=False)

    A_tr, b_tr, _ = build_dataset(os.path.join("data", "stock_train.csv"))
    A_te, b_te, _ = build_dataset(os.path.join("data", "stock_test.csv"))

    res_n = solve_normal_equations(A_tr, b_tr)
    res_g = solve_givens_qr(A_tr, b_tr)
    x = res_g["x"]

    print("=" * 70)
    print("EVALUASI OUT-OF-SAMPLE & INTERPRETASI FINANSIAL (POIN vi & vii)")
    print("=" * 70)

    print("\n[1] RMSE kedua metode")
    print(f"  {'Metode':<18}{'RMSE Train':>14}{'RMSE Test':>14}{'Test/Train':>12}")
    for nama, xm in (("Persamaan Normal", res_n["x"]), ("Givens QR", x)):
        r_tr = calculate_rmse(b_tr, predict(A_tr, xm))
        r_te = calculate_rmse(b_te, predict(A_te, xm))
        print(f"  {nama:<18}{r_tr:>14.6e}{r_te:>14.6e}{r_te / r_tr:>12.3f}")
    d = res_n["x"] - x
    selisih = np.sqrt(np.sum(d * d)) / np.sqrt(np.sum(x * x))
    print(f"  ||x_Normal - x_Givens||_2 / ||x_Givens||_2 = {selisih:.3e}")
    print(f"  residual_norm Normal / Givens = {res_n['residual_norm']:.10e} / {res_g['residual_norm']:.10e}")

    print("\n[2] Pembanding naif (prediksi konstan)")
    for nama, b in (("Train", b_tr), ("Test", b_te)):
        model = calculate_rmse(b, predict(A_tr if nama == "Train" else A_te, x))
        nol = calculate_rmse(b, np.zeros_like(b))
        rata = calculate_rmse(b, np.full_like(b, b_tr.mean()))
        r2 = 1.0 - model ** 2 / np.mean((b - b.mean()) ** 2)
        print(f"  {nama:<5} RMSE model={model:.6e} | prediksi 0={nol:.6e} | "
              f"rata-rata train={rata:.6e} | R^2={r2:.4f} | perbaikan vs nol={(1 - model / nol) * 100:.2f}%")

    print("\n[3] RMSE per rezim (model vs prediksi 0)")
    for nama, A, b in (("Train", A_tr, b_tr), ("Test", A_te, b_te)):
        for rezim, (n, r_model, r_nol) in rmse_per_rezim(A, b, x).items():
            print(f"  {nama:<5} {rezim:<8} n={n:<4} model={r_model:.6e} | nol={r_nol:.6e}")

    print("\n[4] Outlier pada data uji")
    z = (b_te - b_te.mean()) / b_te.std()
    idx = np.where(np.abs(z) > 3.0)[0]
    pred_te = predict(A_te, x)
    print(f"  Target dengan |z| > 3: {[(int(i), round(float(b_te[i]), 6)) for i in idx]}")
    aman = np.abs(z) <= 3.0
    print(f"  RMSE test semua baris         : {calculate_rmse(b_te, pred_te):.6e}")
    print(f"  RMSE test tanpa baris outlier : {calculate_rmse(b_te[aman], pred_te[aman]):.6e}")
    kontribusi = np.sum((b_te[idx] - pred_te[idx]) ** 2) / np.sum((b_te - pred_te) ** 2)
    print(f"  Kontribusi outlier ke jumlah kuadrat galat uji: {kontribusi * 100:.1f}%")

    print("\n[5] Koefisien terbaik (Givens QR) dan signifikansi (data latih)")
    se = standard_errors(A_tr, b_tr, x)
    print(f"  {'Parameter':<10}{'Estimasi':>12}{'Galat baku':>13}{'t':>9}")
    for nama, xj, sj in zip(NAMA_PARAMETER, x, se):
        print(f"  {nama:<10}{xj:>12.6f}{sj:>13.6f}{xj / sj:>9.3f}")

    print("\n[6] Persamaan akhir model")
    print(f"  R_t = {x[0]:+.6f} {x[1]:+.6f} R_(t-1) {x[2]:+.6f} R_(t-2)   jika R_(t-1) >= 0 (Bullish)")
    print(f"  R_t = {x[3]:+.6f} {x[4]:+.6f} R_(t-1) {x[5]:+.6f} R_(t-2)   jika R_(t-1) <  0 (Bearish)")
    print("=" * 70)


if __name__ == "__main__":
    main()
