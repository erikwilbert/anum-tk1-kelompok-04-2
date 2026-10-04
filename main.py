"""
Entry Point Utama Pipeline Eksekusi Tugas Kelompok 1 - Soal 2 (SETAR)
"""

import os
import sys

import numpy as np


def main():
    print("=" * 65)
    print("TK 1 ANALISIS NUMERIK - SOAL 2 (MODEL SETAR)")
    print("Kelompok 04 (Genap): Givens Rotations & Persamaan Normal")
    print("=" * 65)

    train_path = os.path.join("data", "stock_train.csv")
    test_path = os.path.join("data", "stock_test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("\n[PERINGATAN] File dataset belum ditemukan di folder 'data/'.")
        print("Silakan unduh 'stock_train.csv' dan 'stock_test.csv' dari SCeLE")
        print("lalu tempatkan di direktori 'data/'.")
        print("\nUntuk uji coba solver numerik mandiri tanpa CSV, jalankan:")
        print("  python test_solvers.py")
        return

    print("\n[1] Menjalankan Data Pipeline...")
    from src.data_pipeline import build_dataset
    from src.solvers import solve_normal_equations, solve_givens_qr, compute_theoretical_flops
    from src.utils import predict, calculate_rmse
    from src.visual import plot_overlay, plot_overlay_train_test

    A_train, b_train, ret_train = build_dataset(train_path)
    A_test, b_test, ret_test = build_dataset(test_path)
    print(f"  Train: A={A_train.shape}, b={b_train.shape}")
    print(f"  Test : A={A_test.shape}, b={b_test.shape}")

    print("\n[2] Menjalankan Solver Persamaan Normal...")
    res_norm = solve_normal_equations(A_train, b_train)
    print(f"  cond_2(A)     = {res_norm['cond_A']:.4e}")
    print(f"  cond_2(A^T A) = {res_norm['cond_ATA']:.4e}")
    print(f"  residual_norm = {res_norm['residual_norm']:.6e}")
    print(f"  waktu eksekusi = {res_norm['execution_time_ms']:.4f} ms")

    print("\n[3] Menjalankan Solver Givens QR...")
    res_givens = solve_givens_qr(A_train, b_train)
    print(f"  residual_norm = {res_givens['residual_norm']:.6e}")
    print(f"  waktu eksekusi = {res_givens['execution_time_ms']:.4f} ms")

    M, N = A_train.shape
    flops = compute_theoretical_flops(M, N)
    print("\n  Perbandingan FLOPs teoretis (M={}, N={}):".format(M, N))
    print(f"    Persamaan Normal : {flops['normal_flops']:,} FLOPs")
    print(f"    Givens QR        : {flops['givens_flops']:,} FLOPs")

    print("\n[4] Evaluasi Model (Out-of-Sample)...")
    x_opt = res_givens["x"]
    print(f"  x_LS (Givens)  = {x_opt}")
    print(f"  x_Normal       = {res_norm['x']}")
    selisih = x_opt - res_norm["x"]
    selisih_rel = np.sqrt(np.sum(selisih * selisih)) / np.sqrt(np.sum(x_opt * x_opt))
    print(f"  ||x_LS - x_Normal||_2 / ||x_LS||_2 = {selisih_rel:.3e}")

    print(f"\n  {'Metode':<18}{'RMSE Train':>14}{'RMSE Test':>14}")
    for nama, x in (("Persamaan Normal", res_norm["x"]), ("Givens QR", x_opt)):
        rmse_tr = calculate_rmse(b_train, predict(A_train, x))
        rmse_te = calculate_rmse(b_test, predict(A_test, x))
        print(f"  {nama:<18}{rmse_tr:>14.6e}{rmse_te:>14.6e}")

    pred_train = predict(A_train, x_opt)
    pred_test = predict(A_test, x_opt)

    rmse_train = calculate_rmse(b_train, pred_train)
    rmse_test = calculate_rmse(b_test, pred_test)
    print(f"\n  RMSE Data Latih (Train) : {rmse_train:.6e}")
    print(f"  RMSE Data Uji (Test)    : {rmse_test:.6e}")

    print("\n[5] Grafik Overlay...")
    show = "--show" in sys.argv
    path_train = os.path.join("outputs", "overlay_train.png")
    path_train_test = os.path.join("outputs", "overlay_train_test.png")
    plot_overlay(
        b_train, pred_train,
        "Overlay Return Aktual vs Estimasi Model SETAR (Train)",
        save_path=path_train, show=show,
    )
    plot_overlay_train_test(
        b_train, pred_train, b_test, pred_test,
        save_path=path_train_test, show=show,
    )
    print(f"  Disimpan: {path_train}")
    print(f"  Disimpan: {path_train_test}")
    if not show:
        print("  (tambahkan flag --show untuk menampilkan jendela grafik)")


if __name__ == "__main__":
    main()
