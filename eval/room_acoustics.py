import numpy as np

from scipy.io import wavfile
from scipy import stats
from scipy.signal import hilbert, lfilter


from acoustics.signal import bandpass
from acoustics.bands import (_check_band_type, octave_low, octave_high, third_low, third_high)
from acoustics.room import *
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

# detection based on STA/LTA

def align_rir(ir, fs, total_s=4):
    n0 = np.min(np.argmax(np.abs(ir), axis=0)) if ir.ndim > 1 else np.argmax(np.abs(ir))
    out = ir[n0:]
    tgt = int(total_s * fs)
    if len(out) >= tgt:
        out = out[:tgt]
    else:
        pad = tgt - len(out)
        out = np.pad(out, ((0, pad),) if out.ndim == 1 else ((0, pad), (0, 0)))
    return out, n0

def eval_RT(rir, fs, bands, rt='t30'):  # modified from acoustics.room.t60_impulse
    """
    Reverberation time from impulse response signal.

    :param signal: impulse response signal
    :param bands: Octave or third bands as NumPy array [center_freq_low, center_freq_high].
    :param rt: Reverberation time estimator. It accepts `'t30'`, `'t20'`, `'t10'` and `'edt'`.
    :returns: Reverberation time :math:`T_{60}`

    """
    # Assert input is single channel and aligned
    assert rir.ndim == 1, "Input RIR must be single channel"
    signal = rir
    
    band_type = _check_band_type(bands)

    if band_type == 'octave':
        low = octave_low(bands[0], bands[-1])
        high = octave_high(bands[0], bands[-1])
    elif band_type == 'third':
        low = third_low(bands[0], bands[-1])
        high = third_high(bands[0], bands[-1])

    rt = rt.lower()
    if rt == 't30':
        init = -5.0
        end = -35.0
        factor = 2.0
    elif rt == 't20':
        init = -5.0
        end = -25.0
        factor = 3.0
    elif rt == 't10':
        init = -5.0
        end = -15.0
        factor = 6.0
    elif rt == 'edt':
        init = 0.0
        end = -10.0
        factor = 6.0

    t60 = np.zeros(bands.size)

    for band in range(bands.size):
        filtered_signal = bandpass(signal, low[band], high[band], fs)
        max_val = np.max(np.abs(filtered_signal))
        if max_val != 0:
            abs_signal = np.abs(filtered_signal) / max_val
        else:
            abs_signal = np.zeros_like(filtered_signal)

        # Schroeder integration
        sch = np.cumsum(abs_signal[::-1]**2)[::-1]
        sch_db = 10.0 * np.log10(sch / np.max(sch) + 1e-10)

        # Linear regression
        sch_init = sch_db[np.abs(sch_db - init).argmin()]
        sch_end = sch_db[np.abs(sch_db - end).argmin()]
        init_sample = np.where(sch_db == sch_init)[0][0]
        end_sample = np.where(sch_db == sch_end)[0][0]
        x = np.arange(init_sample, end_sample + 1) / fs
        y = sch_db[init_sample:end_sample + 1]
        slope, intercept = stats.linregress(x, y)[0:2]

        # Reverberation time (T30, T20, T10 or EDT)
        db_regress_init = (init - intercept) / slope
        db_regress_end = (end - intercept) / slope
        t60[band] = factor * (db_regress_end - db_regress_init)
    return t60


def eval_clarity(c_time, rir, fs, bands=None): # modified from acoustics.room.clarity
    """
    Clarity :math:`C_i` determined from an impulse response.

    :param time: Time in milliseconds (C50, C80, etc.).
    :param signal: Impulse response.
    :param fs: Sample frequency.
    :param bands: Bands of calculation (optional). Only support standard octave and third-octave bands.

    """
    # Assert input is single channel and aligned
    assert rir.ndim == 1, "Input RIR must be single channel"
    signal = rir
    
    # If no bands provided, return single value
    if bands is None:
        h2 = signal**2.0
        t_samples = int((c_time / 1000.0) * fs + 1)
        return 10.0 * np.log10((np.sum(h2[:t_samples]) / np.sum(h2[t_samples:])))
    
    band_type = _check_band_type(bands)

    if band_type == 'octave':
        low = octave_low(bands[0], bands[-1])
        high = octave_high(bands[0], bands[-1])
    elif band_type == 'third':
        low = third_low(bands[0], bands[-1])
        high = third_high(bands[0], bands[-1])

    c = np.zeros(bands.size)
    for band in range(bands.size):     
        filtered_signal = bandpass(signal, low[band], high[band], fs, order=8)
        h2 = filtered_signal**2.0
        t_samples = int((c_time / 1000.0) * fs + 1)
        c[band] = 10.0 * np.log10((np.sum(h2[:t_samples]) / np.sum(h2[t_samples:])))
    return c

def eval_DRR(rir, fs, direct_win_ms=2.0):
    """
    Direct-to-reverberant ratio from impulse response signal.
    """
    # If stereo, take only one channel
    # Assert input is single channel and aligned
    assert rir.ndim == 1, "Input RIR must be single channel"
    signal = rir
        
    direct_win_samples = int(direct_win_ms * fs / 1000.0)
    direct_signal = signal[:direct_win_samples]
    reverberant_signal = signal[direct_win_samples:]
    return 10.0 * np.log10(np.mean(direct_signal**2) / np.mean(reverberant_signal**2))


def eval_ITDG(rir, gap=16, hold=4, thr_db=-3.0):
    n0   = np.argmax(np.abs(rir))
    print(n0)
    thr  = np.abs(rir[n0]) * 10**(thr_db / 20)
    seg  = np.abs(rir)[n0 + gap:]
    for i in range(len(seg) - hold):
        if np.any(seg[i:i + hold] >= thr):
            return i + gap        # samples after direct peak
    return None




if __name__ == "__main__":

    filepath = "data/RIRS_NOISES/real_rirs_isotropic_noises/RVB2014_type1_rir_largeroom1_far_angla.wav"
    signal, fs = sf.read(filepath)

    signal, n0 = align_rir(signal[:,0], fs)

    # eval RT
    bands = octave(125, 4000)
    rt = eval_RT(signal, fs, bands, rt='t30')
    print(rt)

    # eval clarity
    clarity = eval_clarity(50, signal, fs,bands)
    print(clarity)

    # eval ITDG
    itdg = eval_ITDG(signal, fs)
    print(itdg)


