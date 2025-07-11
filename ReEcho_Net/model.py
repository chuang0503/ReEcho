import torch
import torch.nn as nn
import torchaudio.transforms as T
from icecream import ic
ic.enable()

from modules.rir_modules import *
from modules.wm_modules import *
from speechbrain.inference.vocoders import HIFIGAN


# ─────────────────── transformation blocks ───────────────────
class SpectrogramTransform(nn.Module):
    """
    Convert waveform to complex spectrogram
    """
    def __init__(self, n_fft=1024, win_length=1024, hop_length=256):
        super(SpectrogramTransform, self).__init__()
        self.spec_transform = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=1, # magnitude spectrogram
            normalized=False,
        )
    def forward(self, x): # (batch, channels, time)
        # remove channel dimension
        if x.dim() == 3 and x.size(1) == 1:
          x = x.squeeze(1)
        spec = self.spec_transform(x) #(batch, n_freq, n_time)
        return spec


class MelScaleTransform(nn.Module):
    """
    Convert spectrogram to mel-scale
    """
    def __init__(self, n_fft=1024, win_length=1024, hop_length=256, sample_rate=16000):
        super(MelScaleTransform, self).__init__()
        self.mel_transform = T.MelScale(
            n_mels=80,
            sample_rate=sample_rate,
            n_stft = n_fft // 2 + 1,
            f_min=0,
            f_max=sample_rate/2,
            norm='slaney',
            mel_scale='slaney',
        )

    def log_compression(self, x_mel):
        log_mel = torch.log(torch.clamp(x_mel, min=1e-5))
        return log_mel

    def forward(self, x_spec):
        mel_spec = self.mel_transform(x_spec)
        log_mel_spec = self.log_compression(mel_spec) #(batch, n_mels, n_time)
        return log_mel_spec

# ─────────────────── ReEchoNet ───────────────────
class ReEcho_Separator(nn.Module):
    def __init__(self):
        super(ReEcho_Separator, self).__init__()
        # 1. transformation blocks
        self.spec_transform = SpectrogramTransform()
        self.mel_transform = MelScaleTransform()

        # core modules
        self.conformer_encoder = ConformerEncoder()
        self.masknet = MaskNet_IRM()
        self.rir_embedding = RIREmbedding()

    def forward(self, wav_tensor):
        # 1. spectrogram transform
        spec = self.spec_transform(wav_tensor) # (B, F, T)
        ic(spec.shape, "B, F, T")

        # 2. conformer encoder
        out = self.conformer_encoder(spec) # (B, T, d_model)
        ic(out.shape, "B, T, d_model")

        # 3. masknet
        mask = self.masknet(out) # (B, T, F)
        spec_masked = mask.transpose(1, 2) * spec # (B, F, T)
        ic(spec_masked.shape, "B, F, T")

        # 4. rir
        rir_emb = self.rir_embedding(out) # (B, hidden_dim)

        return spec_masked, rir_emb

class ReEcho_Generator(nn.Module):
    def __init__(self):
        super(ReEcho_Generator, self).__init__()
        self.rir_decoder = RIRDecoder_Decor()
    
    def forward(self, rir_emb):
        # 1. rir generation
        rir_est,_,_ = self.rir_decoder(rir_emb) # (B, 1, T)
        rir_est = F.normalize(rir_est, dim=2, p=2)

        return rir_est

class ReEcho_WM(nn.Module):
    def __init__(self, msg_len=16, hidden_dim=128):
        super(ReEcho_WM, self).__init__()
        self.msg_len = msg_len
        self.hidden_dim = hidden_dim
        self.wm_net = WMEmbedderExtractor(msg_len=msg_len, hidden_dim=hidden_dim)

    def embedding(self, msg, rir_emb):
        emb = self.wm_net.wm_embedding(msg, rir_emb)
        return emb

    def extraction(self, reverb_spec, mode='logit'):
        logit = self.wm_net.wm_extraction(reverb_spec, mode=mode)
        return logit

def test_ReEchoNet():

    wav_tensor = torch.randn(2, 1, 32000)
    separator = ReEcho_Separator()
    generator = ReEcho_Generator()
    # 
    watermarker = ReEcho_WM()
    # 
    spec_masked, rir_emb = separator(wav_tensor)
    msg = torch.randint(0, 2, (2, 16))
    rir_emb_wm = watermarker.embedding(msg, rir_emb)
    rir_est = generator(rir_emb_wm)
    ic(rir_est.shape)
    #
    spec_transform = SpectrogramTransform()
    reverb = torch.randn(2, 1, 16000)
    reverb_spec = spec_transform(reverb)
    msg_logit = watermarker.extraction(reverb_spec, mode='logit')
    ic(msg_logit.shape)
    ic(msg_logit)
    


if __name__ == "__main__":
    test_ReEchoNet()
        