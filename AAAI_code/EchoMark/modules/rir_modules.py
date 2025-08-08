import torch
import torch.nn as nn
import torchaudio.transforms as T
import torch.nn.functional as F

from speechbrain.lobes.models.transformer import Conformer
from speechbrain.nnet.activations import Swish
from speechbrain.nnet.attention import RelPosEncXL
from speechbrain.nnet.pooling import AttentionPooling
from speechbrain.nnet.CNN import ConvTranspose1d, Conv1d
from speechbrain.lobes.models.HifiGAN import HifiganGenerator

from acoustics.bands import octave, octave_low, octave_high
from scipy.signal import firwin
import numpy as np


from icecream import ic
ic.enable()

# --bandpass filters --
def get_bandpass_filters(sr=16000, bp_kernel_size=1023):
    low_freq, high_freq = 20, sr/2
    center_freqs = octave(low_freq, high_freq)
    num_filters = len(center_freqs)
    fir_torch = torch.zeros(num_filters, bp_kernel_size)
    for i, (low,high) in enumerate(zip(octave_low(low_freq, high_freq), octave_high(low_freq, high_freq))):
        fir_torch[i, :] = torch.from_numpy(firwin(bp_kernel_size, [low, min(high, sr/2-100)], window='hamming', fs=sr, pass_zero=False))
    return fir_torch, num_filters

# ─────────────────── Encoder ───────────────────
class ConformerEncoder(nn.Module):
    """
    Conformer encoder
    """
    def __init__(self, input_dim=513, d_model=256, d_ffn=1024, nhead=4, num_layers=12, kernel_size=31, dropout=0.1):
        # try (a) mel input (finished), (b) linear input
        super(ConformerEncoder, self).__init__()

        # input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            Swish()
        )

        # encoder
        self.encoder = Conformer.ConformerEncoder(
            d_model=d_model,
            d_ffn=d_ffn,
            nhead=nhead,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.pos_enc = RelPosEncXL(d_model)

    def forward(self, x_spec):
        # projection
        x_spec_t = x_spec.transpose(1, 2) # (batch, n_time, n_freq)

        # conformer input
        out = self.input_proj(x_spec_t) # （batch, time, feat)
        pos_embs = self.pos_enc(out)
        out, _ = self.encoder(out, pos_embs=pos_embs)
        return out

# ─────────────────── MaskNet ───────────────────
class ResBlock1D_GN(nn.Module):
    def __init__(self, ch: int, groups_gn: int = 8):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=1, bias=False),  # ① point-wise
            nn.GroupNorm(groups_gn, ch),                   #   GN
            Swish(),
            nn.Conv1d(ch, ch, kernel_size=1, bias=False),  # ② point-wise
            nn.GroupNorm(groups_gn, ch),
        )

    def forward(self, x): # (B, D, T)
        return x + self.body(x)

class MaskNet_IRM(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, output_dim=513, depth=3, groups_gn=8):
        super(MaskNet_IRM, self).__init__()
        # 1. input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Swish()
        )
        # 2. res blocks
        self.res_blocks = nn.ModuleList([ResBlock1D_GN(hidden_dim, groups_gn) for _ in range(depth)])

        # 3. output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.mask_act = nn.Sigmoid()

    def forward(self, x_in): # conformer output (B, T, d_model)
        x_in = self.input_proj(x_in).transpose(1, 2) # (B, hidden_dim, T)
        ic(x_in.shape)
        for res_block in self.res_blocks:
            x_in = res_block(x_in)
        x_in = x_in.transpose(1, 2) # (B, T, hidden_dim)
        mask = self.output_proj(x_in) # (B, T, F)
        mask = self.mask_act(mask).transpose(1, 2) # (B, F, T)
        return mask
        
# ─────────────────── RIR extraction ───────────────────
class RIREmbedding(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128):
        super(RIREmbedding, self).__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Swish()
        )
        self.pool = AttentionPooling(input_dim=hidden_dim)
        
    
    def forward(self, x_in):
        x_in = self.input_proj(x_in) # conformer output (B, T, hidden)
        emb = self.pool(x_in) # (B, hidden)
        # emb = F.normalize(emb, p=2, dim=1) # Notice I change here!!!!!

        return emb

class RIRDecoder_Decor(nn.Module):
    def __init__(self, input_dim=128, bp_kernel_size=1023, sr=16000, early_rir_duration=0.05, num_decay=20): # Notice I change here
        super(RIRDecoder_Decor, self).__init__()
        self.sr = sr
        self.early_rir_duration = early_rir_duration
        self.early_rir_length = int(early_rir_duration * sr)
        self.bp_kernel_size = bp_kernel_size
        self.bp_filter_weights, self.num_filters = get_bandpass_filters(sr, bp_kernel_size)
        self.num_decay = num_decay

        # pre conv
        self.conv_pre = ConvTranspose1d(input_dim, 100, in_channels=input_dim, stride=1, padding=0, skip_transpose=True) # transposed !! (B, C, T)!!!


        # early reflection
        self.generator = HifiganGenerator(
            in_channels = input_dim,
            out_channels = 1,
            resblock_type = "1",
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            resblock_kernel_sizes = [3, 7, 11],
            upsample_kernel_sizes = [16, 16, 4, 4],
            upsample_initial_channel = 512,
            upsample_factors = [8, 8, 2, 2],
        )

        # late reverb noise shaping
        self.filter = nn.Conv1d(
            self.num_filters,
            self.num_filters,
            kernel_size=1023,
            stride=1,
            padding='same',
            groups=self.num_filters,
            bias=False,
        )
        self.filter.weight.data = self.bp_filter_weights.unsqueeze(1)
        self.filter.weight.requires_grad = True

        # shared backbone MLP
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            Swish(),
            nn.GroupNorm(8, 256),

            nn.Linear(256, 256),
            Swish(),
            nn.GroupNorm(8, 256),

            nn.Linear(256, 512),
            Swish(),
            nn.GroupNorm(8, 512),
        )
        self.amplitude_head = nn.Sequential(
            nn.Linear(512, 200),
            Swish(),
            nn.Linear(200, self.num_filters * self.num_decay),
        )

        self.gate_head = nn.Sequential(
            nn.Linear(512, 200),
            Swish(),
            nn.Linear(200, self.num_filters * self.num_decay),
            nn.Sigmoid()
        )
        self.tau_max = 3.0/6.91 # tau = T60 / Ln(10^3)
        self.tau_seed = nn.Parameter(torch.randn(self.num_decay))

    def forward(self, rir_emb, noise=None): # require (B, input_dim)

        # pre conv
        seed = self.conv_pre(rir_emb.unsqueeze(-1)) #(B, hidden, T=100)

        decay_tau = (F.sigmoid(self.tau_seed) * self.tau_max).view(1, self.num_decay, 1) # (1, num_decay, 1)

        out = self.generator(seed) # (B, 1, T)
        B, _, T = out.shape
        ic(out.shape)

        # 1. direct path and early RIR
        rir = torch.zeros_like(out)
        early = out[:, :, :self.early_rir_length]
        rir[:, :, :self.early_rir_length] += early # （B, 1, T)

        # exponential decay
        emb = self.backbone(rir_emb)
        amplitude = self.amplitude_head(emb).view(-1, self.num_filters, self.num_decay)
        gate = self.gate_head(emb).view(-1, self.num_filters, self.num_decay)
        amplitude = amplitude * gate # (B, num_filters, num_decay)
        
        t_end = (T-self.early_rir_length)/self.sr
        t_vec = torch.arange(0, t_end, 1/self.sr, device=out.device).detach()
        t_vec = t_vec.view(1, 1, -1).repeat(B, self.num_decay, 1) # (B, num_decay, T)
        envelope = torch.exp(-t_vec / decay_tau) # (B, num_decay, T)
        ic(amplitude.shape, envelope.shape, t_end)


        # noise shaping
        late_reverb_freq = torch.matmul(amplitude, envelope) # (B, num_filters, T)
        ic(late_reverb_freq.shape)

        if noise is None:
            noise = torch.randn_like(late_reverb_freq).detach()
        filtered_noise = self.filter(noise) # (B, num_filters, T)
        ic(filtered_noise.shape)

        late_reverb = late_reverb_freq * filtered_noise

        # superpose
        rir[:, :, self.early_rir_length:] += late_reverb.sum(dim=1, keepdim=True) # (B, 1, T)

        
        return rir, late_reverb_freq, filtered_noise

class RIRDecoder_Fins(nn.Module):
    def __init__(self, input_dim=128, bp_kernel_size=1023, sr=16000, early_rir_duration=0.05):
        super(RIRDecoder_Fins, self).__init__()
        self.sr = sr
        self.early_rir_duration = early_rir_duration
        self.early_rir_length = int(early_rir_duration * sr)
        self.bp_kernel_size = bp_kernel_size
        self.bp_filter_weights, self.num_filters = get_bandpass_filters(sr, bp_kernel_size)
        ic(self.num_filters)

        # pre conv
        self.conv_pre = ConvTranspose1d(input_dim, 100, in_channels=input_dim, stride=1, padding=0, skip_transpose=True) # transposed !! (B, C, T)!!!

        self.generator = HifiganGenerator(
            in_channels = input_dim,
            out_channels = self.num_filters + 1,
            resblock_type = "1",
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            resblock_kernel_sizes = [3, 7, 11],
            upsample_kernel_sizes = [16, 16, 4, 4],
            upsample_initial_channel = 512,
            upsample_factors = [8, 8, 2, 2],
        )

        self.generator.conv_post = Conv1d(
            in_channels=32,
            out_channels= self.num_filters + 1,
            kernel_size=7,
            stride=1,
            padding="same",
            skip_transpose=True,
            bias=True,
            weight_norm=True,
        )

        self.filter = nn.Conv1d(
            self.num_filters,
            self.num_filters,
            kernel_size=1023,
            stride=1,
            padding='same',
            groups=self.num_filters,
            bias=False,
        )
        self.filter.weight.data = self.bp_filter_weights.unsqueeze(1)
        self.filter.weight.requires_grad = True

    def forward(self, rir_emb, noise=None): # require B, C, T
        # pre conv
        seed = self.conv_pre(rir_emb.unsqueeze(-1)) #(B, hidden, T=100)
        out = self.generator(seed) # (B, N, T)
        ic(out.shape)
        # 1. direct path and early RIR
        early = torch.zeros_like(out[:, 0, :])
        early[:, :self.early_rir_length] = out[:, 0, :self.early_rir_length]

        # 2. late RIR
        if noise is None:
            noise = torch.randn_like(out[:, 1:, :]).detach()
        ic(noise.shape)
        mask = torch.sigmoid(out[:, 1:, :])
        filtered_noise = self.filter(noise) * mask
        late = out[:, 1:, :] * filtered_noise

        # superpose
        rir_ch = torch.cat([early.unsqueeze(1), late], dim=1) # (B, N+1, T)
        rir = rir_ch.sum(dim=1, keepdim=True) # (B, 1, T)
        
        return rir, rir_ch, filtered_noise
        
# ─────────────────── test ───────────────────
def test_decor():
    rir_decoder = RIRDecoder_Decor(input_dim=128, sr=16000, early_rir_duration=0.05)
    rir_emb = torch.randn(2, 128)

    rir, late_reverb_freq, filtered_noise = rir_decoder(rir_emb)
    print(rir.shape, late_reverb_freq.shape, filtered_noise.shape)

def test_masknet():
    masknet = MaskNet_IRM(input_dim=256, hidden_dim=256, output_dim=513)
    x_in = torch.randn(2, 100, 256)
    mask_cat = masknet(x_in)
    print(mask_cat.shape)


   
    
        
if __name__ == "__main__":
    test_masknet()
    test_decor()




