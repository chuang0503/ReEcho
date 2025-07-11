import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.nnet.activations import Swish


from icecream import ic
ic.enable()

# ------------------- watermarking -------------------
class WMEmbedderExtractor(nn.Module):
    def __init__(self, msg_len=16, bps=4, hidden_dim=128):
        super(WMEmbedderExtractor, self).__init__()
        self.msg_len = msg_len
        self.bps = bps
        self.weight = 2** torch.arange(bps-1, -1, -1)
        self.msg_codebook = nn.Embedding(2**bps+1, hidden_dim) # last is SOS token
        self.n = msg_len // bps
        # embedder
        self.embedder_init = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim*2),
            Swish(),
            nn.LayerNorm(hidden_dim*2)
        )
        self.embedder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.LayerNorm(hidden_dim)
        )
        self.seq_embedder = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)

       # extractor
        self.extractor_init = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim*2),
            Swish(),
            nn.LayerNorm(hidden_dim*2)
        )
        self.extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.LayerNorm(hidden_dim)
        )
        self.seq_extractor = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.extractor_proj = nn.Linear(hidden_dim, 2**bps)

    
    def msg_to_idx(self, msg):
        # message to token (B, msg_len) => (B, N)
        B, _ = msg.shape
        msg_chip = msg.reshape(B, self.n, self.bps)
        w = self.weight.to(msg.device)
        msg_token_idx = (msg_chip * w).sum(dim=2).long()
        return msg_token_idx
    
    def idx_to_onehot(self, msg_token_idx):
        # token index to one-hot (B, N) => (B, N, 2**bps)
        B, N = msg_token_idx.shape
        device = msg_token_idx.device
        one_hot = F.one_hot(msg_token_idx, num_classes=2**self.bps).float() # (B, N, 2**bps)
        return one_hot
    
    def idx_to_msg(self, msg_token_idx):
        # token index to message (B, N) => (B, msg_len)
        B, N = msg_token_idx.shape
        device = msg_token_idx.device

        # bit
        bit_pos = torch.arange(self.bps - 1, -1, -1, device=device)
        bits = ((msg_token_idx.unsqueeze(-1) >> bit_pos) & 1).long()
        msg = bits.reshape(B, N * self.bps)
        return msg
    
    def wm_embedding(self, msg_token_idx, emb):
        emb = emb.unsqueeze(1) # (B, 1, hidden_dim)
        # tokenize
        msg_token = self.msg_codebook(msg_token_idx) # (B, N, hidden_dim)
        # init hidden state
        sos_token = self.msg_codebook(torch.tensor([2**self.bps], device=emb.device)) # (1,hidden_dim)
        ic(sos_token.shape)
        sos_token = sos_token.unsqueeze(0).repeat(emb.shape[0], 1, 1) # (B, 1, hidden_dim)
        h_and_c = self.embedder_init(torch.cat([emb, sos_token], dim=-1)).transpose(0, 1) # (1, B, hidden_dim*2)
        h, c = h_and_c.split(h_and_c.shape[-1]//2, dim=-1) # (1, B, hidden_dim) to fit lstm
        h = h.contiguous()
        c = c.contiguous()
        # embedding
        x_in = self.embedder(msg_token) # (B, N, hidden_dim) batch first
        out, _ = self.seq_embedder(x_in, (h, c)) # (B, N, hidden_dim)
        emb = out[:, -1, :] # (B, hidden_dim)
        emb = F.normalize(emb, dim=-1, p=2)
        return emb

    def wm_extraction(self, emb, mode='train', tau=1, t_force=0.5, msg_idx=None):
        if mode == 'train':
            teacher_forcing = torch.rand(self.n) < t_force
            assert msg_idx is not None, "msg_idx is required for teacher forcing"
        else:
            teacher_forcing = False
        # extractor
        weight = self.msg_codebook.weight # (2**bps+1, hidden_dim)
        # init hidden state
        emb = emb.unsqueeze(1) # (B, 1, hidden_dim)
        sos_token = self.msg_codebook(torch.tensor([2**self.bps], device=emb.device)) # (1, hidden_dim)
        sos_token = sos_token.unsqueeze(0).repeat(emb.shape[0], 1, 1) # (B, 1, hidden_dim)
        h_and_c = self.extractor_init(torch.cat([emb, sos_token], dim=-1)).transpose(0, 1) # (1, B, hidden_dim*2)
        h, c = h_and_c.split(h_and_c.shape[-1]//2, dim=-1) # (1, B, hidden_dim) to fit lstm
        h = h.contiguous()
        c = c.contiguous()
        ic(h.shape, c.shape)
        # extraction
        x_in = self.extractor(sos_token) # (B, 1, hidden_dim)
        pred_list = []
        for i in range(self.n):
            out, (h, c) = self.seq_extractor(x_in, (h, c)) # (B, 1, hidden_dim)
            out = self.extractor_proj(out) # (B, 1, 2**bps)
            pred_list.append(out) # (B, 1, 2**bps)
            # gumbel-softmax
            if mode == 'train':
                if teacher_forcing[i]:
                    token_next_pred = self.msg_codebook(msg_idx[:, i]).unsqueeze(1) # (B, 1, hidden_dim)
                    ic("teacher forcing", token_next_pred.shape)
                else:
                    out = F.gumbel_softmax(out, tau=tau, hard=True) # (B, 1, 2**bps)
                    token_next_pred = torch.matmul(out, weight[0:-1].detach()) # (B, 1, hidden_dim)
                    ic("gumbel-softmax", token_next_pred.shape)
            else:
                token_next_pred = self.msg_codebook(torch.argmax(out, dim=-1)) # (B, 1, hidden_dim)
                ic("eval", token_next_pred.shape)
            x_in = self.extractor(token_next_pred) # (B, 1, hidden_dim)

        pred_logit = torch.cat(pred_list, dim=1) # (B, N, 2**bps)
        return pred_logit


def test_wm():
    rir_emb = torch.randn(2, 128)
    msg = torch.randint(0, 2, (2, 32))
    ic(msg)

    wm = WMEmbedderExtractor(msg_len=32, bps=4, hidden_dim=128)

    msg_idx = wm.msg_to_idx(msg)
    ic(msg_idx)
    msg_onehot = wm.idx_to_onehot(msg_idx)
    ic(msg_onehot)

    # embedding
    emb = wm.wm_embedding(msg_idx, rir_emb)
    ic(emb.shape)

    # extraction (train)
    msg_logit = wm.wm_extraction(emb, mode='train', msg_idx=msg_idx, tau=1, t_force=1)
    ic(msg_logit.shape)

    msg_logit = wm.wm_extraction(emb, mode='train', msg_idx=msg_idx, tau=1, t_force=0)
    ic(msg_logit.shape)

    # extraction (test)
    msg_logit = wm.wm_extraction(emb, mode='test', msg_idx=None, tau=1, t_force=0)
    ic(msg_logit.shape)

    # test msg_idx
    idx = msg_logit.argmax(dim=-1)
    msg = wm.idx_to_msg(idx)
    ic(msg_logit, idx, msg)

if __name__ == "__main__":
    test_wm()