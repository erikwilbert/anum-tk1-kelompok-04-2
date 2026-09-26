"""
Pengujian Solver Numerik Soal 2 (Model SETAR)
Penanggung Jawab: Person A (Erik Wilbert)
Metode: Givens Rotations & Persamaan Normal
"""

import numpy as np
from src.solvers import (
    solve_normal_equations,
    solve_givens_qr,
    get_first_givens_elimination,
    compute_theoretical_flops
)
from src.utils import predict, calculate_rmse


def test_first_givens_on_toy():
    print("=" * 70)
    print("1. PENGUJIAN VERIFIKASI LANGKAH AWAL GIVENS ROTATION (POIN iv)")
    print("   Menggunakan Matriks Ilustrasi Halaman 10 PDF")
    print("=" * 70)

    A_toy = np.array([
        [0.0,  0.0000,  0.0000, 1.0, -0.0198, +0.0100],
        [1.0, +0.0152, -0.0198, 0.0,  0.0000,  0.0000],
        [0.0,  0.0000,  0.0000, 1.0, -0.0050, +0.0152]
    ], dtype=np.float64)

    print(f"Matriks A awal (ukuran {A_toy.shape}):")
    print(A_toy)

    c, s, G1, G1_A = get_first_givens_elimination(A_toy)

    print(f"\nParameter rotasi Givens pengeliminasi A[1, 0]:")
    print(f"  cos(theta) = {c:.6f}")
    print(f"  sin(theta) = {s:.6f}")
    print("\nMatriks Rotasi G1 (ukuran 3x3):")
    print(np.round(G1, 6))

    print("\nHasil perkalian G1 @ A:")
    print(np.round(G1_A, 6))

    assert abs(G1_A[1, 0]) < 1e-14, f"Elemen sub-diagonal pertama gagal dieliminasi! Nilai = {G1_A[1, 0]}"
    print("\n[VERIFIKASI SUKSES]: Elemen G1_A[1, 0] = 0.000000 (tereliminasi secara sempurna).")


def test_overdetermined_system():
    print("\n" + "=" * 70)
    print("2. PENGUJIAN SOLVER OVERDETERMINED (PERSAMAAN NORMAL vs GIVENS QR)")
    print("   Menggunakan Sistem Uji Overdetermined Sintetis (M=12, N=6)")
    print("=" * 70)

    np.random.seed(42)
    m_samples = 20
    n_params = 6

    A_synth = np.zeros((m_samples, n_params), dtype=np.float64)
    returns = np.random.normal(loc=0.001, scale=0.015, size=m_samples + 2)

    for idx, t in enumerate(range(2, len(returns))):
        r_prev1 = returns[t - 1]
        r_prev2 = returns[t - 2]
        is_bullish = 1.0 if r_prev1 >= 0 else 0.0

        A_synth[idx, 0] = is_bullish
        A_synth[idx, 1] = is_bullish * r_prev1
        A_synth[idx, 2] = is_bullish * r_prev2
        A_synth[idx, 3] = 1.0 - is_bullish
        A_synth[idx, 4] = (1.0 - is_bullish) * r_prev1
        A_synth[idx, 5] = (1.0 - is_bullish) * r_prev2

    b_synth = returns[2:]

    print(f"Dimensi Matriks Desain A : {A_synth.shape}")
    print(f"Dimensi Vektor Target b  : {b_synth.shape}")

    res_normal = solve_normal_equations(A_synth, b_synth)
    res_givens = solve_givens_qr(A_synth, b_synth)
    x_numpy_ref, _, _, _ = np.linalg.lstsq(A_synth, b_synth, rcond=None)

    print("\n--- HASIL ESTIMASI PARAMETER x ---")
    param_names = ["alpha_1", "phi_11", "phi_12", "alpha_2", "phi_21", "phi_22"]
    header = f"{'Parameter':<12} | {'Persamaan Normal':<18} | {'Givens QR':<18} | {'NumPy Ref (LSTSQ)':<18}"
    print(header)
    print("-" * len(header))
    for i in range(n_params):
        print(f"{param_names[i]:<12} | {res_normal['x'][i]:<18.6e} | {res_givens['x'][i]:<18.6e} | {x_numpy_ref[i]:<18.6e}")

    diff_norm = np.linalg.norm(res_givens['x'] - x_numpy_ref)
    assert diff_norm < 1e-10, f"Selisih solusi Givens QR terlalu besar: {diff_norm}"
    print(f"\n[VALIDASI SUKSES]: Selisih Givens QR vs NumPy LSTSQ = {diff_norm:.2e} (Cocok sempurna!)")

    print("\n--- ANALISIS KESTABILAN NUMERIK & CONDITION NUMBER (POIN iii) ---")
    print(f"Condition Number kappa_2(A)     : {res_normal['cond_A']:.4e}")
    print(f"Condition Number kappa_2(A^T A) : {res_normal['cond_ATA']:.4e}")
    ratio = res_normal['cond_ATA'] / (res_normal['cond_A'] ** 2)
    print(f"Rasio kappa_2(A^T A) / (kappa_2(A))^2: {ratio:.4f} (Membuktikan pengkuadratan condition number)")

    print("\n--- ANALISIS RESIDUAL & KINERJA (POIN v) ---")
    print(f"Norm Residual ||A x_Normal - b||_2 : {res_normal['residual_norm']:.6e}")
    print(f"Norm Residual ||A x_Givens - b||_2 : {res_givens['residual_norm']:.6e}")
    print(f"Waktu Eksekusi Persamaan Normal    : {res_normal['execution_time_ms']:.4f} ms")
    print(f"Waktu Eksekusi Givens QR           : {res_givens['execution_time_ms']:.4f} ms")

    pred_givens = predict(A_synth, res_givens['x'])
    rmse_val = calculate_rmse(b_synth, pred_givens)
    print(f"Root Mean Square Error (RMSE)      : {rmse_val:.6e}")


def test_flops_and_memory():
    print("\n" + "=" * 70)
    print("3. PERBANDINGAN FLOPS TEORETIS & MEMORI (POIN v)")
    print("   Simulasi Ukuran Dataset Riil: M = 300 observasi, N = 6 parameter")
    print("=" * 70)

    perf = compute_theoretical_flops(M=300, N=6)
    print(f"Observasi M = {perf['M']}, Parameter N = {perf['N']}")
    print(f"- FLOPs Teoretis Persamaan Normal : {perf['normal_flops']:,} FLOPs")
    print(f"- FLOPs Teoretis Givens QR        : {perf['givens_flops']:,} FLOPs")
    print(f"- Konsumsi Memori Tambahan Normal : {perf['normal_mem_bytes']} bytes (Simpan A^T A & A^T b)")
    print(f"- Konsumsi Memori Givens Implisit : {perf['givens_mem_implicit_bytes']} bytes (In-place)")
    print(f"- Konsumsi Memori Givens Eksplisit: {perf['givens_mem_explicit_bytes']:,} bytes (Jika simpan matriks Q eksplisit 300x300)")
    print("=" * 70)


if __name__ == "__main__":
    test_first_givens_on_toy()
    test_overdetermined_system()
    test_flops_and_memory()
