import numpy as np


def predict(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Menghitung prediksi return saham:
    R_hat = A @ x
    """
    return A @ x


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Menghitung Root Mean Square Error (RMSE) sesuai spesifikasi Poin vi:
    RMSE = sqrt( (1/N) * sum_{t=1}^N (R_t - R_hat_t)^2 )
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
