import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

def plot_spectrogram_wav(
    wav: np.ndarray,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 256,
    title: str = "Spectrogram",
    cmap: str = "inferno",
    figsize: tuple = (3.5, 2.5),
):
    if wav.ndim == 2:
        wav = wav.mean(0)

    f, t, Zxx = stft(wav, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
    S_db = 10 * np.log10(np.abs(Zxx) ** 2 + 1e-10)

    plt.figure(figsize=figsize, dpi=300)
    im = plt.pcolormesh(t, f / 1000, S_db, shading="gouraud", cmap=cmap)
    plt.title(title, fontsize=9)
    plt.xlabel("Time (s)", fontsize=8)
    plt.ylabel("Frequency (kHz)", fontsize=8)
    cbar = plt.colorbar(im, pad=0.015, format="%+0.1f dB")
    cbar.ax.tick_params(labelsize=6)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout(pad=0.1)
    plt.show()

def plot_waveform(
    wav: np.ndarray,
    sr: int = 16000,
    title: str = "Waveform",
    figsize: tuple = (3.5, 1.8),
    color: str = "tab:blue",
):
    if wav.ndim == 2:
        wav = wav.mean(0)
    t = np.arange(len(wav)) / sr
    plt.figure(figsize=figsize, dpi=300)
    plt.plot(t, wav, linewidth=0.8, color=color)
    plt.title(title, fontsize=9)
    plt.xlabel("Time (s)", fontsize=8)
    plt.ylabel("Amplitude", fontsize=8)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.ylim(-1, 1)
    plt.tight_layout(pad=0.1)
    plt.show()
