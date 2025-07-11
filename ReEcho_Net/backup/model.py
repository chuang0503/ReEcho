import torch
import torch.nn as nn
import torchaudio.transforms as T
import torch.nn.functional as F

from speechbrain.lobes.models.transformer import Conformer
from speechbrain.nnet.activations import Swish
from speechbrain.nnet.attention import RelPosEncXL
from speechbrain.nnet.pooling import AttentionPooling
from speechbrain.nnet.CNN import ConvTranspose1d



from icecream import ic
ic.enable()

# initial-stage: wave to spectrogram
# net 1: wav-spec to mask
# net 2: feature to waveform
# post-stage: spectrogram to mel-scale

# ─────────────────── transformation blocks ───────────────────
class SpectrogramTransform(nn.Module):
    """
    Convert waveform to spectrogram
    """
    def __init__(self, n_fft=1024, win_length=1024, hop_length=256):
        super(SpectrogramTransform, self).__init__()
        self.spec_transform = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=1,
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
        out, attn_list = self.encoder(out, pos_embs=pos_embs)
        ic(out.shape)
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

    def forward(self, x):
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
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    def forward(self, x_in): # conformer output (B, T, d_model)
        x_in = self.input_proj(x_in).transpose(1, 2) # (B, hidden_dim, T)
        for res_block in self.res_blocks:
            x_in = res_block(x_in)
        m_proj = self.output_proj(x_in.transpose(1, 2)) # (B, T, F)
        mask = m_proj.transpose(1, 2) # (B, F, T)
        return mask
        
# ─────────────────── RIR extraction ───────────────────
class RIREmbedding(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, kernel_size=16):
        super(RIREmbedding, self).__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Swish()
        )
        self.pool = AttentionPooling(input_dim=hidden_dim)
        self.conv_last = ConvTranspose1d(hidden_dim, kernel_size, in_channels=hidden_dim, stride=1, padding=0, skip_transpose=True) # transposed !! (B, C, T)!!!
    
    def forward(self, x_in):
        x_in = self.input_proj(x_in) # conformer output (B, T, hidden)
        emb = self.pool(x_in) # (B, hidden)
        emb = F.normalize(emb, p=2, dim=1)
        seed = self.conv_last(emb.unsqueeze(2)) #(B, hidden, T=16)

        return emb, seed

class UpSampleLayer(nn.Module):
    def __init__(self, out_channels, kernel_size, in_channels, stride):
        super(UpSampleLayer, self).__init__()
        self.conv_layers = nn.ModuleList([
            ConvTranspose1d(out_channels, kernel_size, in_channels=in_channels, stride=stride, padding=(kernel_size-stride)//2, skip_transpose=True), # transposed !! (B, C, T)!!!
            nn.GroupNorm(8, out_channels), 
            Swish()
        ])
    def forward(self, x_in): # 
        for layer in self.conv_layers:
            x_in = layer(x_in) # in/out (B, C, T)
        return x_in

class UpSampleDecoder(nn.Module):
    def __init__(self, kernel_size=41, upsample_factor=[8, 4, 4, 2], channels=[128, 64, 32, 16, 16]):
        super(UpSampleDecoder, self).__init__()
        # upsample blocks
        self.up_blocks = nn.ModuleList([
            UpSampleLayer(channels[i+1], kernel_size, channels[i], stride=s) for i, s in enumerate(upsample_factor)
        ])
        # 3. output projection
        self.conv_last = ConvTranspose1d(1, kernel_size, in_channels=channels[-1], stride=1, padding=0, skip_transpose=True)
        self.tanh = nn.Tanh()
    
    def forward(self, seed):
        out = seed
        for up_block in self.up_blocks:
            out = up_block(out) # B, C, T
            ic(out.shape)

        out = self.conv_last(out)
        ic(out.shape)

        x_out = self.tanh(out) # B, C=1, T
        ic(x_out.shape)

        return x_out

# early rir estimator
class EarlyRIREstimator(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_layers=3, num_reflection=80, sr=16000, early_rir_duration=0.05):
        super(EarlyRIREstimator, self).__init__()
        self.early_rir_duration = early_rir_duration
        self.early_rir_length = int(early_rir_duration * sr)
        self.num_reflection = num_reflection

        #
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Swish()
        )
        layers = [nn.Linear(hidden_dim, hidden_dim), Swish(), nn.BatchNorm1d(hidden_dim)]
        for _ in range(num_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim), Swish(), nn.BatchNorm1d(hidden_dim)]
        self.backbone = nn.Sequential(*layers)

        self.score_proj = nn.Linear(hidden_dim, self.early_rir_length)
        self.amp_proj = nn.Linear(hidden_dim, self.early_rir_length)
        self.amp_act = nn.Tanh()
    
    def soft_topk(self, scores: torch.Tensor, k: int, tau: float = 0.1):
        B, N = scores.shape
        # 1) pairwise differences (B, N, N)
        diff = scores.unsqueeze(2) - scores.unsqueeze(1)          # s_i - s_j
        A    = torch.sigmoid(diff / tau)                          # soft indicator

        # 2) soft ranks  r_i = 1 + sum_j indicator(s_j > s_i)
        r = 1 + A.sum(dim=1)                                      # (B, N)

        # 3) probability of being in Top-k
        p = torch.sigmoid((k + 0.5 - r) / tau)                    # (B, N)
        return p
        
    def forward(self, emb):
        emb = self.input_proj(emb) # (B, hidden)
        emb = self.backbone(emb)

        scores = self.score_proj(emb)
        p = self.soft_topk(scores, k=self.num_reflection) # (B, T)

        amp = self.amp_act(self.amp_proj(emb)) # (B, T)
        return p, amp
    
    def to_waveform_train(self, p, amp, sigma=1.0):
        B, T = p.shape
        device = p.device
        impulses = p * amp                   # (B, T)

        return impulses.unsqueeze(1)

    def to_waveform_test(self, p, amp):
        B, T = p.shape
        device = p.device
        
        # Sample from Bernoulli distribution with probability p
        bernoulli = torch.bernoulli(p)
        rir = (bernoulli * amp).unsqueeze(1)  # (B, 1, T)
        return rir
    
# ─────────────────── test ───────────────────
def test_conformer_encoder():
    torch.manual_seed(0)
    B, T_sec, sr, d_model = 2, 1.0, 16_000, 256
    wav = torch.randn(B, 1, int(T_sec*sr))       # dummy data

    spec_tf = SpectrogramTransform()
    mel_tf  = MelScaleTransform()
    enc     = ConformerEncoder(d_model=d_model)

    with torch.no_grad():
        spec = spec_tf(wav)                      # (B, F, T)
        # mel  = mel_tf(spec)                     # (B, 80, T)
        out = enc(spec)                    # forward (B, T, d_model)

    # ---- assert shape ----
    assert out.shape[0] == B                    # batch
    assert out.shape[1] == spec.shape[2]         # time length
    assert out.shape[2] == d_model              # d_model
    print("✓ ConformerEncoder test passed!")

def test_IRM():
    # ─────────── model params ───────────
    B, T = 4, 160        # batch, time
    D_in  = 256          # conformer d_model
    H_mid = 256          # hidden_dim
    F_out = 513          # frequency bin
    depth = 6            # depth

    # ─────────── build model ───────────
    net = MaskNet_IRM(
        input_dim=D_in,
        hidden_dim=H_mid,
        output_dim=F_out,
        depth=depth,
        groups_gn=8,
    ).eval()

    # ─────────── input ───────────
    x = torch.randn(B, T, D_in)  # dummy data

    with torch.no_grad():
        mask = net(x)            # (B, T, F)

    # ─────────── assert shape ───────────
    assert mask.shape == (B, F_out, T), "shape mismatch"

    print("✓ test_IRM passed — shape", mask.shape,
          ", min %.4f  max %.4f" % (mask.min(), mask.max()))


def test_RIRDecoder():
    # ─────────── model params ───────────
    B, T = 4, 160        # batch, time
    D_in  = 256          # conformer d_model
    H_mid = 128          # hidden_dim
    
    rir_emb = RIREmbedding(
        input_dim=D_in,
        hidden_dim=H_mid,
    )
    rir_dec = UpSampleDecoder()
    
    # ─────────── input ───────────
    x = torch.randn(B, T, D_in)  # dummy data
    
    with torch.no_grad():
        rir_emb, seed = rir_emb(x)
        ic(rir_emb.shape, seed.shape)
        rir_dec = rir_dec(seed)

def test_early_reflect(batch=4, emb_dim=128):
    """Quick sanity-check for EarlyRIREstimator outputs."""

    # 1.embedding
    emb = torch.randn(batch, emb_dim)

    # 2. 
    model = EarlyRIREstimator(
        input_dim     = emb_dim,
        hidden_dim    = 128,
        num_layers    = 3,
        num_reflection= 10,          # k
        sr            = 16_000,
        early_rir_duration = 0.05    # 50 ms → T = 800
    )

    # 3. forward
    p, amp = model(emb)                 # (B, T)
    T = p.size(1)

    # 4. generate early RIR
    rir_train = model.to_waveform_train(p, amp)  # (B, 1, T)
    rir_test  = model.to_waveform_test(p, amp)         # (B, 1, T)

    # 5. 
    assert p.shape       == (batch, T)
    assert amp.shape     == (batch, T)
    assert rir_train.shape == (batch, 1, T)
    assert rir_test.shape  == (batch, 1, T)

    #   probability should be in [0,1]
    assert torch.all(p >= 0) and torch.all(p <= 1)
    #   tanh output should be in [-1,1]
    assert torch.all(torch.abs(amp) <= 1)

    #   non-zero impulse count in inference mode ≤ k
    nz = [(s != 0).sum().item() for s in rir_test]
    assert all(n <= model.num_reflection for n in nz)

    print("✓ EarlyRIREstimator passes shape & value checks!")
    print(f"   train RIR shape {rir_train.shape}, "
          f"non-zeros per sample {nz}")
        
if __name__ == "__main__":
    test_conformer_encoder()
    test_IRM()
    test_early_reflect()



