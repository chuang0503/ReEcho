import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.nnet.activations import Swish
import torchaudio.transforms as T

from speechbrain.lobes.models.transformer import Conformer
from speechbrain.nnet.attention import RelPosEncXL
from speechbrain.nnet.pooling import AttentionPooling

from icecream import ic
ic.enable()

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

class RIREmbedding(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, kernel_size=100):
        super(RIREmbedding, self).__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Swish()
        )
        self.pool = AttentionPooling(input_dim=hidden_dim)
        
    
    def forward(self, x_in):
        x_in = self.input_proj(x_in) # conformer output (B, T, hidden)
        emb = self.pool(x_in) # (B, hidden)
        # emb = F.normalize(emb, p=2, dim=1)

        return emb

# ------------------- watermarking -------------------
class WMEmbedderExtractor(nn.Module):
    def __init__(self, msg_len=5, hidden_dim=128):
        super(WMEmbedderExtractor, self).__init__()
        self.msg_len = msg_len
        self.weight = 2** torch.arange(msg_len-1, -1, -1)
        self.msg_codebook = nn.Embedding(2**msg_len, hidden_dim)
        # embedder

        self.embedder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.LayerNorm(hidden_dim),
        )

       # extractor
        # input projection
        self.encoder = ConformerEncoder(input_dim=513, d_model=256, d_ffn=1024, nhead=4, num_layers=12, kernel_size=31, dropout=0.1)
        self.pooling = RIREmbedding(input_dim=256, hidden_dim=128)
        self.extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, msg_len+1) # +1 for non-watermarked
        )
    
    def msg_to_idx(self, msg):
        # message to token (B, msg_len) => (B, )
        B, _ = msg.shape
        w = self.weight.to(msg.device)
        msg_token_idx = (msg * w).sum(dim=1).long()
        return msg_token_idx
    
    def idx_to_msg(self, msg_token_idx):
        # token index to message (B, ) => (B, msg_len)
        device = msg_token_idx.device
        msg_token_idx = msg_token_idx.unsqueeze(1)


        # bit
        bit_pos = torch.arange(self.msg_len - 1, -1, -1, device=device)
        bits = ((msg_token_idx >> bit_pos) & 1).long()
        bits = bits.squeeze(1)
        return bits
    
    def wm_embedding(self, msg, emb):
        msg_idx = self.msg_to_idx(msg) # (B, )
        msg_emb = self.msg_codebook(msg_idx) # (B, hidden_dim)
        # fuze 
        emb = self.embedder(msg_emb) + emb # (B, hidden_dim)
        # emb = F.normalize(emb, dim=-1, p=2)
        return emb

    def wm_extraction(self, reverb_spec, mode='train'):
        emb_T = self.encoder(reverb_spec)
        emb = self.pooling(emb_T)
        tmp = self.extractor(emb) # (B, msg_len)
        pred_logit = tmp[:, :-1] # (B, msg_len)
        wm_exist = tmp[:, -1] # (B, )
        if mode == 'train':
            wm_exist, out = wm_exist, pred_logit
        elif mode == 'test':
            wm_exist = (wm_exist>0.0).float()
            out = (pred_logit>0.0).float() # (B, msg_len)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        return wm_exist, out


def test_wm():
    rir_emb = torch.randn(2, 128)
    msg = torch.randint(0, 2, (2, 16))
    ic(msg)

    wm = WMEmbedderExtractor(msg_len=16,hidden_dim=128)

    msg_idx = wm.msg_to_idx(msg)
    ic(msg_idx)
    msg_dec = wm.idx_to_msg(msg_idx)
    ic(msg_dec)

    # embedding
    emb = wm.wm_embedding(msg, rir_emb)
    ic(emb.shape)

    # extraction (train)
    reverb = torch.randn(2, 1, 16000)
    spec_transform = SpectrogramTransform()
    reverb_spec = spec_transform(reverb)
    exist, msg_out = wm.wm_extraction(reverb_spec, mode='train')
    ic(msg_out)

    # extraction (test)
    exist, msg_out = wm.wm_extraction(reverb_spec, mode='test')
    ic(exist)
    ic(msg_out)

    # ber
    ber = ((msg_out>0.5).float() != msg).float().mean()
    ic(ber)


if __name__ == "__main__":
    test_wm()