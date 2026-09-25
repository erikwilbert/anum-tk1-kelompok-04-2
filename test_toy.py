"""
Sandbox Pengujian Solver Menggunakan Contoh Toy Problem (Halaman 10 PDF)
Digunakan oleh Person A (Erik) untuk menguji solver sebelum dataset asli selesai diproses.
"""

import numpy as np

# Matriks Desain A dan Vektor Target b dari Ilustrasi Halaman 10 PDF
A_toy = np.array([
    [0.0,  0.0000,  0.0000, 1.0, -0.0198, +0.0100],
    [1.0, +0.0152, -0.0198, 0.0,  0.0000,  0.0000],
    [0.0,  0.0000,  0.0000, 1.0, -0.0050, +0.0152]
], dtype=np.float64)

b_toy = np.array([+0.0152, -0.0050, +0.0200], dtype=np.float64)

if __name__ == "__main__":
    print("=" * 60)
    print("UJI COBA MATRIKS TOY (HALAMAN 10 PDF)")
    print("=" * 60)
    print(f"Dimensi A_toy: {A_toy.shape}")
    print(f"Dimensi b_toy: {b_toy.shape}")
    print("\nMatriks A_toy:")
    print(A_toy)
    print("\nVektor b_toy:")
    print(b_toy)
    print("\nStatus: Siap dihubungkan dengan solver di src/solvers.py")
