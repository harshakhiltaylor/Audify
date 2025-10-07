import numpy as np
import librosa
from scipy.signal import butter, lfilter

# Audio constants
SR = 16000
N_FFT = 512
HOP_LENGTH = 128
WINDOW_TYPE = 'hann'


def butter_lowpass_filter(data: np.ndarray, cutoff: float, sr: int = SR, order: int = 6) -> np.ndarray:
    """
    Apply a low-pass Butterworth filter to the audio data.
    """
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)


def extract_features(file_path: str, sr: int = SR):
    """
    Load an audio file and compute its magnitude spectrogram in dB.

    Returns:
        db_feats (np.ndarray): Array of shape (frames, freq_bins) in dB.
        waveform (np.ndarray): Raw audio signal.
        stft (np.ndarray): Complex STFT matrix (freq_bins, frames).
    """
    y, _ = librosa.load(file_path, sr=sr)
    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, window=WINDOW_TYPE)
    mag = np.abs(stft)
    db = librosa.amplitude_to_db(mag)
    return db.T, y, stft