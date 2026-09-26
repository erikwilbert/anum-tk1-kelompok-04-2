"""
Entry Point Utama Pipeline Eksekusi Tugas Kelompok 1 - Soal 2 (SETAR)
"""

import os
import sys


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
    from src.solvers import solve_normal_equations, solve_givens_qr
    from src.utils import predict, calculate_rmse

    A_train, b_train, ret_train = build_dataset(train_path)
    A_test, b_test, ret_test = build_dataset(test_path)
    print(f"  Train: A={A_train.shape}, b={b_train.shape}")
    print(f"  Test : A={A_test.shape}, b={b_test.shape}")

    print("\n[2] Menjalankan Solver Persamaan Normal...")
    res_norm = solve_normal_equations(A_train, b_train)

    print("\n[3] Menjalankan Solver Givens QR...")
    res_givens = solve_givens_qr(A_train, b_train)

    print("\n[4] Evaluasi Model (Out-of-Sample)...")
    x_opt = res_givens["x"]
    pred_train = predict(A_train, x_opt)
    pred_test = predict(A_test, x_opt)

    rmse_train = calculate_rmse(b_train, pred_train)
    rmse_test = calculate_rmse(b_test, pred_test)

    print(f"  RMSE Data Latih (Train) : {rmse_train:.6e}")
    print(f"  RMSE Data Uji (Test)    : {rmse_test:.6e}")


if __name__ == "__main__":
    main()
