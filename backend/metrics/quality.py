import numpy as np
from pesq import pesq
from pystoi.stoi import stoi


FRAME_LEN = 512
OVERLAP = 256
SR = 16000


def segmental_snr(clean: np.ndarray, enhanced: np.ndarray, frame_len: int = FRAME_LEN, overlap: int = OVERLAP) -> float:
    """
    Compute segmental SNR between clean and enhanced signals.
    """
    eps = 1e-10
    clean = clean[:len(enhanced)]
    snr_vals = []
    step = frame_len - overlap
    for i in range(0, len(clean) - frame_len, step):
        c = clean[i:i+frame_len]
        e = enhanced[i:i+frame_len]
        noise = c - e
        signal_energy = np.sum(c**2)
        noise_energy = np.sum(noise**2) + eps
        if signal_energy > 0:
            snr_vals.append(10 * np.log10(signal_energy / noise_energy))
    return float(np.mean(snr_vals))


def compute_pesq(clean: np.ndarray, enhanced: np.ndarray, sr: int = SR) -> float:
    """
    Wideband PESQ score.
    """
    length = min(len(clean), len(enhanced))
    return pesq(sr, clean[:length], enhanced[:length], 'wb')


def compute_stoi(clean: np.ndarray, enhanced: np.ndarray, sr: int = SR) -> float:
    """
    STOI intelligibility metric.
    """
    length = min(len(clean), len(enhanced))
    return stoi(clean[:length], enhanced[:length], sr, extended=False)