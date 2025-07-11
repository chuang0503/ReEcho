import numpy as np
import torch
from scipy import stats
from acoustics.signal import bandpass
from acoustics.bands import (_check_band_type, octave_low, octave_high, third_low, third_high)
from acoustics.bands import octave
import soundfile as sf

# Target RIR metrics:
# 1. [AVRIR/RIR-Compress]T60 (t10/20/30/edt) done
# 2. [RIR-Compress] Clarity (C50, C80, etc.) done
# 3. [RIR-Compress] Energy decay curve (EDC)
# 4. [RIR-Compress] echo density
# 5. [AVRIR] DRR (direct-to-reverberant ratio) done
# 6. [AVRIR] EMSE/LMSE (how to define early/late component of RIR) TO Do
# 7. ITDG (Initial Time Delay Gap) done

SOUNDSPEED = 343.0

# -------------------------------------------------------------
# Helper utilities
# -------------------------------------------------------------

def _to_numpy(x):
    """Return *x* as a NumPy array without modifying the original object."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _dispatch_over_batch(func):
    """Decorator that enables *func* (expecting a single-channel 1‑D signal) to
    transparently accept tensors of shape *(B, C, T)*, *(C, T)*, or *(T,)*.

    The wrapped function returns results stacked along the *batch* and *channel*
    dimensions so that the user can directly call it with either NumPy arrays
    or PyTorch tensors without changing existing code.
    """
    def wrapper(sig, *args, **kwargs):
        sig_np = _to_numpy(sig)

        # (T,) – single signal -------------------------------------------------
        if sig_np.ndim == 1:
            return func(sig_np, *args, **kwargs)

        # (C, T) – multi‑channel, no batch ------------------------------------
        if sig_np.ndim == 2:
            outs = [func(ch, *args, **kwargs) for ch in sig_np]
            return np.stack(outs, axis=0)

        # (B, C, T) – batch + channel -----------------------------------------
        if sig_np.ndim == 3:
            outs = [[func(ch, *args, **kwargs) for ch in ex] for ex in sig_np]
            return np.stack([np.stack(ex, axis=0) for ex in outs], axis=0)

        raise ValueError(
            f"Unsupported input with {sig_np.ndim} dimensions. Expected 1D, 2D, or 3D array.")
    return wrapper

# -------------------------------------------------------------
# Core metrics (now tensor‑friendly)
# -------------------------------------------------------------

def align_rir(ir, fs, total_s: float = 2):
    """Align **ir** on its first sample above the peak envelope.

    :param ir: 1‑D RIR (``np.ndarray`` or ``torch.Tensor``) **or** tensor with
               shape *(C, T)* or *(B, C, T)*.
    :param fs: Sampling rate (Hz).
    :param total_s: Target total seconds after alignment (default **4** s).
    :returns: ``(aligned_rir, n0)`` where *n0* is the sample index (or array of
              indices for batched input) of the detected direct sound.
    """
    ir_np = _to_numpy(ir)

    # ---- single impulse (T,) ----------------------------------------------
    if ir_np.ndim == 1:
        n0 = int(np.argmax(np.abs(ir_np)))
        out = ir_np[n0:] / ir_np[n0]
        tgt_len = int(total_s * fs)
        if len(out) >= tgt_len:
            out = out[:tgt_len]
        else:
            out = np.pad(out, (0, tgt_len - len(out)))
        return out, n0

    # ---- multi‑channel (C, T) ---------------------------------------------
    if ir_np.ndim == 2:
        aligned, indices = zip(*(align_rir(ch, fs, total_s) for ch in ir_np))
        return np.stack(aligned, axis=0), np.array(indices)

    # ---- batch + channel (B, C, T) ----------------------------------------
    if ir_np.ndim == 3:
        aligned, indices = zip(*(align_rir(ex, fs, total_s) for ex in ir_np))
        return np.stack(aligned, axis=0), np.stack(indices, axis=0)

    raise ValueError("Input must be 1D, 2D or 3D array/tensor.")


@_dispatch_over_batch
def eval_RT(rir, fs, bands, rt: str = "t30"):  # noqa: N802
    """Reverberation time from impulse response signal.

    <Original docstring unchanged>
    """
    assert rir.ndim == 1, "Input RIR must be single channel"
    signal = rir  # alias to keep original variable name

    band_type = _check_band_type(bands)

    if band_type == "octave":
        low = octave_low(bands[0], bands[-1])
        high = octave_high(bands[0], bands[-1])
    elif band_type == "third":
        low = third_low(bands[0], bands[-1])
        high = third_high(bands[0], bands[-1])

    rt = rt.lower()
    if rt == "t30":
        init, end, factor = -5.0, -35.0, 2.0
    elif rt == "t20":
        init, end, factor = -5.0, -25.0, 3.0
    elif rt == "t10":
        init, end, factor = -5.0, -15.0, 6.0
    elif rt == "edt":
        init, end, factor = 0.0, -10.0, 6.0
    else:
        raise ValueError(f"Unknown RT estimator '{rt}'")

    t60 = np.zeros(bands.size, dtype=np.float64)

    for b in range(bands.size):
        filtered = bandpass(signal, low[b], high[b], fs)
        max_val = np.max(np.abs(filtered))
        abs_sig = np.abs(filtered) / max_val if max_val else np.zeros_like(filtered)

        sch = np.cumsum(abs_sig[::-1] ** 2)[::-1]
        sch_db = 10.0 * np.log10(sch / np.max(sch) + 1e-10)

        init_sample = np.abs(sch_db - init).argmin()
        end_sample = np.abs(sch_db - end).argmin()
        x = np.arange(init_sample, end_sample + 1) / fs
        y = sch_db[init_sample:end_sample + 1]
        slope, intercept, *_ = stats.linregress(x, y)

        t60[b] = factor * ((end - intercept) / slope - (init - intercept) / slope)

    return t60


@_dispatch_over_batch
def eval_clarity(rir, fs, c_time=50, bands=None):  # noqa: N802
    """Clarity :math:`C_i` determined from an impulse response.

    <Original docstring unchanged>
    """
    assert rir.ndim == 1, "Input RIR must be single channel"
    signal = rir

    if bands is None:
        h2 = signal ** 2.0
        t_samples = int((c_time / 1000.0) * fs + 1)
        return 10.0 * np.log10(np.sum(h2[:t_samples]) / np.sum(h2[t_samples:]))

    band_type = _check_band_type(bands)
    if band_type == "octave":
        low = octave_low(bands[0], bands[-1])
        high = octave_high(bands[0], bands[-1])
    elif band_type == "third":
        low = third_low(bands[0], bands[-1])
        high = third_high(bands[0], bands[-1])

    c = np.zeros(bands.size, dtype=np.float64)
    for b in range(bands.size):
        filt = bandpass(signal, low[b], high[b], fs, order=8)
        h2 = filt ** 2.0
        t_samples = int((c_time / 1000.0) * fs + 1)
        c[b] = 10.0 * np.log10(np.sum(h2[:t_samples]) / np.sum(h2[t_samples:]))
    return c


@_dispatch_over_batch
def eval_DRR(rir, fs, direct_win_ms: float = 2.0):  # noqa: N802
    """Direct‑to‑reverberant ratio from impulse response signal."""
    assert rir.ndim == 1, "Input RIR must be single channel"
    signal = rir

    win = int(direct_win_ms * fs / 1000.0)
    direct, rev = signal[:win], signal[win:]
    return 10.0 * np.log10(np.mean(direct ** 2) / np.mean(rev ** 2))


@_dispatch_over_batch
def eval_ITDG(rir, gap: int = 16, hold: int = 4, thr_db: float = -3.0):  # noqa: N802
    """Initial Time Delay Gap (ITDG)."""
    n0 = int(np.argmax(np.abs(rir)))
    thr = np.abs(rir[n0]) * 10 ** (thr_db / 20)
    seg = np.abs(rir)[n0 + gap:]
    for i in range(len(seg) - hold):
        if np.any(seg[i:i + hold] >= thr):
            return i + gap  # samples after direct peak
    return None


# -------------------------------------------------------------
# Example usage
# -------------------------------------------------------------
if __name__ == "__main__":
    filepath = "/home/c3-server1/Documents/Chenpei/re-rir/data/rirs_noises/RIRS_NOISES/real_rirs_isotropic_noises/RVB2014_type1_rir_largeroom1_far_angla.wav"
    signal, fs = sf.read(filepath)

    # Convert to (B, C, T) tensor (batch size 1)
    signal_t = torch.from_numpy(signal.T).unsqueeze(0)  # (1, 2, T)

    aligned, n0 = align_rir(signal_t, fs)

    bands = octave(125, 4000)
    rt = eval_RT(aligned[:, 0], fs, bands)  # evaluate only first channel
    print("T30 per octave:", rt)

    clarity = eval_clarity(aligned[:, 0], fs, bands=bands)
    print("C50 per octave:", clarity)

    itdg = eval_ITDG(aligned[:, 0])
    print("ITDG:", itdg)
