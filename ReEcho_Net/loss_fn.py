
import torch.nn as nn
import torch
from torchaudio.transforms import Spectrogram
from icecream import ic
import torch.nn.functional as F

# spectral loss
class MSSTFT_Loss(nn.Module):
    def __init__(self, 
        n_fft=[1024, 512, 256],
        win_length=[1024, 512, 256],
        hop_length=[512, 256, 128],
        power=1.0,
        weights=(1, 1)):

        super(MSSTFT_Loss, self).__init__()
        
        multi_scale = []
        for n_fft, win_length, hop_length in zip(n_fft, win_length, hop_length):
            multi_scale.append(
                Spectrogram(
                    n_fft=n_fft,
                    win_length=win_length,
                    hop_length=hop_length,
                    power=power
                )
            )
        self.multi_scale = nn.ModuleList(multi_scale)
        self.weights = weights

    def log_compression(self, x):
        y = torch.log(torch.clamp(x, min=1e-5))
        return y

    def forward(self, predicted, target):
        # Clip predicted and target to the same (shorter) length along the last dimension
        min_len = min(predicted.shape[-1], target.shape[-1])
        predicted = predicted[..., :min_len]
        target = target[..., :min_len]
        # normalize
        predicted = F.normalize(predicted, dim=2, p=2)
        target = F.normalize(target, dim=2, p=2)
        # compute loss
        log_mag_loss, spec_conv_loss = 0, 0
        for spec_transform in self.multi_scale:
            predicted_spec = spec_transform(predicted).squeeze(1)
            target_spec = spec_transform(target).squeeze(1)
            ic(predicted_spec.shape, target_spec.shape)
            
            # convergence loss
            diff = predicted_spec - target_spec
            fro_diff = torch.linalg.norm(diff, ord="fro",dim=(-2,-1))
            fro_target = torch.linalg.norm(target_spec, ord="fro",dim=(-2,-1))
            spec_conv_loss += (fro_diff / (fro_target + 1e-6)).mean()

            # log magnitude loss
            log_predicted_spec = self.log_compression(predicted_spec)
            log_target_spec = self.log_compression(target_spec)
            log_mag_diff = log_predicted_spec - log_target_spec
            log_mag_loss += log_mag_diff.abs().mean()
            ic(log_mag_loss, spec_conv_loss)

        loss = log_mag_loss*self.weights[0] + spec_conv_loss*self.weights[1]


        return loss

class STFT_Loss(nn.Module):
    def __init__(self, weights=(1, 1)):
        super(STFT_Loss, self).__init__()
        self.weights = weights

    def log_compression(self, x):
        y = torch.log(torch.clamp(x, min=1e-5))
        return y

    def forward(self, predicted, target):
        # Clip predicted and target to the same (shorter) length along the last dimension
        min_len = min(predicted.shape[-1], target.shape[-1])
        predicted = predicted[..., :min_len]
        target = target[..., :min_len]

        # compute loss
        log_mag_loss, spec_conv_loss = 0, 0
        predicted_spec = predicted.squeeze(1) # (B, F, T)
        target_spec = target.squeeze(1) # (B, F, T)
        ic(predicted_spec.shape, target_spec.shape)
            
        # convergence loss
        diff = predicted_spec - target_spec
        fro_diff = torch.linalg.norm(diff, ord="fro",dim=(-2,-1))
        fro_target = torch.linalg.norm(target_spec, ord="fro",dim=(-2,-1))
        spec_conv_loss += (fro_diff / (fro_target + 1e-6)).mean()

        # log magnitude loss
        log_predicted_spec = self.log_compression(predicted_spec)
        log_target_spec = self.log_compression(target_spec)
        log_mag_diff = log_predicted_spec - log_target_spec
        log_mag_loss += log_mag_diff.abs().mean()
        ic(log_mag_loss, spec_conv_loss)

        loss = log_mag_loss*self.weights[0] + spec_conv_loss*self.weights[1]


        return loss

# watermark loss NOMA
# class WM_Loss(nn.Module):
#     def __init__(self, msg_len=16, bps=4):
#         super(WM_Loss, self).__init__()
#         self.N = msg_len // bps
#         self.weights = 2 ** torch.arange(self.N-1, -1, -1)
    
#     def forward(self, msg_logit, onehots):
#         assert self.N == msg_logit.shape[1], "msg_logit and onehots must have the same N"
#         loss = 0
#         for i,weight in enumerate(self.weights):
#             pred = msg_logit[:, i, :] # (B, 2**bps)
#             gt = onehots[:,i,:]
#             loss += F.cross_entropy(pred, gt) * weight
#         return loss

class WM_Loss(nn.Module):
    def __init__(self, msg_len=16):
        super(WM_Loss, self).__init__()
        self.bce_logit = nn.BCEWithLogitsLoss()
    
    def forward(self, msg_logit, msg):
        loss = self.bce_logit(msg_logit, msg.float())
        return loss

if __name__ == "__main__":
    predicted = torch.randn(4, 1, 16555)
    target = torch.randn(4, 1, 16000)
    loss_msstft = MSSTFT_Loss()
    print(loss_msstft(predicted, target).item())

    predicted = torch.randn(4, 513, 100)
    target = torch.randn(4, 513, 99)
    loss_stft = STFT_Loss()
    print(loss_stft(predicted, target).item())

    msg_logit = torch.randn(4, 16)
    msg = torch.randint(0, 2, (4, 16), dtype=torch.float32)
    loss_wm = WM_Loss(msg_len=16)
    print(loss_wm(msg_logit, msg).item())