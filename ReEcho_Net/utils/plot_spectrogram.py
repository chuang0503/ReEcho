import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────
# 1. 直接给波形 -> 先算频谱再画
# ──────────────────────────────────────────────
def plot_spectrogram_wav(
    wav: torch.Tensor,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 256,
    title: str = "Spectrogram",
    cmap: str = "inferno",
    figsize: tuple = (3.5, 2.5),
):
    """
    wav : 1‑D [T] 或 2‑D [C, T] 的张量，支持 GPU / CPU
    """
    # 1) 波形维度整理
    if wav.dim() == 2:
        wav = wav.mean(0, keepdim=True)   # 多声道取均值；若想保留声道请自行修改
    elif wav.dim() == 1:
        wav = wav.unsqueeze(0)            # (T) -> (1, T)

    # 2) 计算功率谱
    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft, hop_length=hop_length, power=2.0
    )
    with torch.no_grad():
        spec = spec_transform(wav.cpu())  # -> [1, F, T]
    spec = spec.squeeze(0)                # [F, T]

    # 3) 画图
    _plot_spectrogram_tensor(
        spec, sr, hop_length, title, cmap, figsize
    )


# ──────────────────────────────────────────────
# 2. 已有频谱张量 -> 直接画
# ──────────────────────────────────────────────
def plot_spectrogram_spec(
    spec: torch.Tensor,
    sr: int = 16000,
    hop_length: int = 256,
    title: str = "Spectrogram",
    cmap: str = "inferno",
    figsize: tuple = (3.5, 2.5),
):
    """
    spec : [F, T] 或 [1, F, T] 的功率谱张量（线性幅 / 功率皆可）
    """
    if spec.dim() == 3 and spec.size(0) == 1:
        spec = spec.squeeze(0)

    _plot_spectrogram_tensor(
        spec.cpu(), sr, hop_length, title, cmap, figsize
    )


# ──────────────────────────────────────────────
# 内部公共绘图函数
# ──────────────────────────────────────────────
def _plot_spectrogram_tensor(
    spec: torch.Tensor,
    sr: int,
    hop_length: int,
    title: str,
    cmap: str,
    figsize: tuple,
):
    # 转 NumPy & dB
    S_db = 10 * np.log10(spec.numpy() + 1e-10)

    # 生成坐标
    time = np.arange(S_db.shape[1]) * hop_length / sr          # s
    freq = np.linspace(0, sr / 2, S_db.shape[0]) / 1000        # kHz

    # 画图
    plt.figure(figsize=figsize, dpi=300)                       # 300 dpi 确保清晰
    im = plt.pcolormesh(time, freq, S_db,
                        shading="gouraud", cmap=cmap)

    plt.title(title, fontsize=9)
    plt.xlabel("Time (s)", fontsize=8)
    plt.ylabel("Frequency (kHz)", fontsize=8)

    cbar = plt.colorbar(im, pad=0.015, format="%+0.1f dB")
    cbar.ax.tick_params(labelsize=6)

    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout(pad=0.1)
    plt.show()


# ──────────────────────────────────────────────
# 3. 画时域波形
# ──────────────────────────────────────────────
def plot_waveform(
    wav: torch.Tensor,
    sr: int = 16000,
    title: str = "Waveform",
    figsize: tuple = (3.5, 1.8),   # 稍矮一点
    color: str = "tab:blue",
):
    """
    wav : 1‑D [T] 或 2‑D [C, T] Tensor；支持 GPU / CPU
    """
    # 单声道化并转 NumPy
    if wav.dim() == 2:
        wav = wav.mean(0)          # 若想单独画每个声道，可循环调用
    wav_np = wav.cpu().numpy()

    # 时间坐标
    t = np.arange(len(wav_np)) / sr

    # 画图
    plt.figure(figsize=figsize, dpi=300)
    plt.plot(t, wav_np, linewidth=0.8, color=color)
    plt.title(title, fontsize=9)
    plt.xlabel("Time (s)", fontsize=8)
    plt.ylabel("Amplitude", fontsize=8)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.ylim(-1, 1)
    plt.tight_layout(pad=0.1)
    plt.show()

def plot_waveform_with_vline(
    wav: torch.Tensor,
    sr: int = 16000,
    vline_ms: float = 50.0,
    title: str = "Waveform",
    figsize: tuple = (3.5, 1.8),
    color: str = "tab:blue",
):
    if wav.dim() == 2:
        wav = wav.mean(0)
    wav_np = wav.cpu().numpy()
    t = np.arange(len(wav_np)) / sr           # 时间坐标 (s)

    plt.figure(figsize=figsize, dpi=300)
    plt.plot(t, wav_np, linewidth=0.8, color=color)
    plt.axvline(x=vline_ms / 1000.0,                # 50 ms → 0.05 s
                linestyle="--", linewidth=0.8,
                color="tab:red", label=f"{vline_ms} ms")
    plt.title(title, fontsize=9)
    plt.xlabel("Time (s)", fontsize=8)
    plt.ylabel("Amplitude", fontsize=8)
    plt.legend(frameon=False, fontsize=7)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.ylim(-1, 1)
    plt.tight_layout(pad=0.1)
    plt.show()