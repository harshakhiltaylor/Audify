import os
import io
import json
import time
from flask import Flask, request, jsonify

import numpy as np
import librosa
import soundfile as sf

from tensorflow.keras.models import Sequential, load_model as _load_model
from tensorflow.keras.layers import Dense, Input

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.features import extract_features, butter_lowpass_filter, HOP_LENGTH, WINDOW_TYPE, SR
from metrics.quality import segmental_snr, compute_pesq, compute_stoi

# Paths for saving/loading
MODEL_PATH = "models/frame_model.keras"
STATS_PATH = "models/norm_stats.json"


def build_frame_model(input_dim: int) -> Sequential:
    """
    Build and compile a simple frame-wise DNN model.
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(1024, activation="relu"),
        Dense(input_dim, activation="linear")
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def prepare_data(clean_path: str, noisy_path: str):
    """
    Load matching clean/noisy files, extract features, normalize frames.
    Returns normalized X, Y, and normalization stats.
    """
    X_frames, Y_frames = [], []
    for fname in os.listdir(clean_path):
        clean_file = os.path.join(clean_path, fname)
        noisy_file = os.path.join(noisy_path, fname)
        if os.path.exists(noisy_file):
            clean_feats, _, _ = extract_features(clean_file)
            noisy_feats, _, _ = extract_features(noisy_file)
            min_len = min(clean_feats.shape[0], noisy_feats.shape[0])
            X_frames.append(noisy_feats[:min_len])
            Y_frames.append(clean_feats[:min_len])

    X = np.vstack(X_frames)
    Y = np.vstack(Y_frames)
    mean = np.mean(X)
    std = np.std(X)
    X_norm = (X - mean) / std
    Y_norm = (Y - mean) / std
    return X_norm, Y_norm, mean, std


def train_model(clean_path: str, noisy_path: str, epochs: int = 10, batch_size: int = 32):
    """
    Train frame model on the given dataset.
    Saves the trained model + stats before returning.
    """
    X, Y, mean, std = prepare_data(clean_path, noisy_path)
    model = build_frame_model(X.shape[1])
    history = model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=1)
    
    # Save trained artifacts
    save_trained_model(model, mean, std)
    return model, mean, std


def enhance_audio(model, noisy_file: str, mean: float, std: float,
                  output_path: str = None,
                  output_buffer: io.BytesIO = None,
                  update_progress=None, processing_id=None):
    """
    Enhance a single noisy audio file, save output, and report metrics.
    """
    for p in range(0, 10, 1):
            update_progress(processing_id, p)
            time.sleep(0.05)
        
    feats, y_noisy, stft_noisy = extract_features(noisy_file)
    norm_feats = (feats - mean) / std
    pred = model.predict(norm_feats, verbose=0)
    pred = (pred * std) + mean

    for p in range(10, 70, 1):
            update_progress(processing_id, p)
            time.sleep(0.05)
    # Reconstruct waveform
    mag = librosa.db_to_amplitude(pred.T)
    phase = np.angle(stft_noisy[:, :mag.shape[1]])
    enhanced_stft = mag * np.exp(1j * phase)
    enhanced = librosa.istft(enhanced_stft, hop_length=HOP_LENGTH, window=WINDOW_TYPE)
    enhanced = butter_lowpass_filter(enhanced, cutoff=4000, sr=SR)

    for p in range(70, 80, 5):
            update_progress(processing_id, p)
            time.sleep(0.05)
    # Output to file or in-memory buffer
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, enhanced, SR)
    if output_buffer is not None:
        sf.write(output_buffer, enhanced, SR, format="WAV")

    for p in range(80, 99, 5):
            update_progress(processing_id, p)
            time.sleep(0.05)
    # Calculate and return metrics
    seg = segmental_snr(y_noisy, enhanced)
    p = compute_pesq(y_noisy, enhanced)
    s = compute_stoi(y_noisy, enhanced)

    update_progress(processing_id, 99)
    
    return seg, p, s


def save_trained_model(model, mean, std):
    """
    Save model (in native Keras format) and normalization stats (JSON).
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)  
    stats = {"mean": float(mean), "std": float(std)}
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f)


def load_trained_model():
    """
    Load model and stats if they exist, else return (None, None, None).
    """
    if os.path.exists(MODEL_PATH) and os.path.exists(STATS_PATH):
        model = _load_model(MODEL_PATH)
        stats = json.load(open(STATS_PATH))
        return model, stats["mean"], stats["std"]
    return None, None, None