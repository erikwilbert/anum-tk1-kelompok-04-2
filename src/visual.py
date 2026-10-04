"""
visual.py — Grafik Overlay Return Aktual vs Prediksi Model SETAR
Penanggung Jawab: Person B (Data & Financial Modeling)
Sub-Poin: vii
"""

import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

# --- Konstanta tampilan (satu sumber kebenaran untuk styling seluruh modul) ---
COLOR_ACTUAL = "#1f77b4"
COLOR_PREDICTED = "#d62728"
LABEL_ACTUAL = "Return Aktual"
LABEL_PREDICTED = "Return Estimasi (SETAR)"
DPI = 150


def _draw_overlay(ax: plt.Axes, t: np.ndarray, actual: np.ndarray, predicted: np.ndarray) -> None:
    """Menggambar sepasang garis overlay (aktual vs prediksi) ke satu axes."""
    ax.plot(t, actual, label=LABEL_ACTUAL, color=COLOR_ACTUAL, linewidth=1.2)
    ax.plot(t, predicted, label=LABEL_PREDICTED, color=COLOR_PREDICTED,
            linewidth=1.2, linestyle="--")


def _finalize(ax: plt.Axes, fig: plt.Figure, title: str, xlabel: str,
              save_path: Optional[str], legend_loc: str = "upper right",
              show: bool = True) -> None:
    """
    Menerapkan styling umum (judul, label, grid, legenda).

    Menyimpan gambar ke file hanya jika save_path diberikan (tidak None).
    Menampilkan jendela plot secara langsung jika show=True (default).
    Jika save_path=None dan show=False, gambar dibuang begitu saja tanpa efek.
    """
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Return")
    ax.legend(loc=legend_loc, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=DPI)

    if show:
        plt.show()
    plt.close(fig)


def plot_overlay(actual: np.ndarray, predicted: np.ndarray, title: str,
                  save_path: Optional[str] = None, show: bool = True) -> None:
    """
    Membuat plot time series tunggal yang menumpangkan (overlay) return aktual
    dengan return hasil estimasi model SETAR.

    Parameters:
    -----------
    actual : np.ndarray, shape=(M,)
        Return aktual R_t (target b dari build_dataset).
    predicted : np.ndarray, shape=(M,)
        Return hasil prediksi model (A @ x_LS).
    title : str
        Judul grafik.
    save_path : str, optional
        Jika diisi, gambar disimpan ke path tersebut (mis. 'overlay_train.png').
        Jika None (default), gambar TIDAK disimpan ke disk sama sekali.
    show : bool, default=True
        Jika True, tampilkan jendela plot secara langsung.
    """
    fig, ax = plt.subplots(figsize=(11, 4.5))
    t = np.arange(1, len(actual) + 1)

    _draw_overlay(ax, t, actual, predicted)
    _finalize(ax, fig, title, "Indeks Waktu (t)", save_path, show=show)


def plot_overlay_train_test(
    actual_train: np.ndarray,
    predicted_train: np.ndarray,
    actual_test: np.ndarray,
    predicted_test: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Membuat plot time series kontinu yang menggabungkan segmen Train dan Test,
    menumpangkan return aktual dengan return hasil estimasi model SETAR,
    disertai penanda visual batas antara kedua segmen.

    Parameters:
    -----------
    actual_train, predicted_train : np.ndarray, shape=(M_train,)
        Return aktual dan prediksi pada data latih.
    actual_test, predicted_test : np.ndarray, shape=(M_test,)
        Return aktual dan prediksi pada data uji.
    save_path : str, optional
        Jika diisi, gambar disimpan ke path tersebut. Jika None (default),
        gambar TIDAK disimpan ke disk sama sekali.
    show : bool, default=True
        Jika True, tampilkan jendela plot secara langsung.
    """
    actual = np.concatenate([actual_train, actual_test])
    predicted = np.concatenate([predicted_train, predicted_test])
    n_train, n_total = len(actual_train), len(actual)
    t = np.arange(1, n_total + 1)

    fig, ax = plt.subplots(figsize=(13, 5))
    _draw_overlay(ax, t, actual, predicted)

    boundary = n_train + 0.5
    ax.axvline(x=boundary, color="black", linestyle=":", linewidth=1.5)
    ax.axvspan(boundary, n_total, color="gray", alpha=0.08)

    y_bottom = ax.get_ylim()[0]
    ax.text(n_train / 2, y_bottom * 0.92, "TRAIN", ha="center", fontsize=10, fontweight="bold")
    ax.text(boundary + (n_total - n_train) / 2, y_bottom * 0.92, "TEST",
            ha="center", fontsize=10, fontweight="bold")

    _finalize(
        ax, fig,
        "Overlay Return Aktual vs Estimasi Model SETAR (Train + Test)",
        "Indeks Waktu (t, kontinu Train\u2192Test)",
        save_path,
        legend_loc="upper left",
        show=show,
    )