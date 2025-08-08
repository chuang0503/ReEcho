
import torch.nn as nn
import torch
from torchaudio.transforms import Spectrogram
from icecream import ic
import torch.nn.functional as F
from asteroid.losses.sdr import SingleSrcNegSDR

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
        # # normalize # I cancle it because it will cause problems
        # predicted = F.normalize(predicted, dim=2, p=2)
        # target = F.normalize(target, dim=2, p=2)
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

class EarlySDR_Loss(nn.Module):
    def __init__(self, early_reflection = 0.05, sr=16000, mode="sdsdr"):
        super(EarlySDR_Loss, self).__init__()
        self.length = int(early_reflection * sr)
        self.sdr = SingleSrcNegSDR(sdr_type=mode, take_log=True, reduction="mean")


    def forward(self, predicted, target):
        # clip predicted and target to the same length
        predicted = predicted[..., :self.length].squeeze(1)
        target = target[..., :self.length].squeeze(1)
        # compute loss
        loss = self.sdr(predicted, target)
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

class EDCLoss(nn.Module):
    """
    Differentiable EDC loss.
    mode='lin'  → NMSE on linear EDC
    mode='log'  → L1 on log10(EDC)
    """
    def __init__(self, mode: str = "log", reduction: str = "mean",
                 eps: float = 1e-8):
        super().__init__()
        assert mode in ("lin", "log")
        assert reduction in ("mean", "sum", "none")
        self.mode      = mode
        self.reduction = reduction
        self.eps       = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target : [B, T] or [B, 1, T]
        返回 loss (标量 or batch 维)
        """
        pred   = pred.squeeze(1) if pred.dim() == 3 else pred
        target = target.squeeze(1) if target.dim() == 3 else target

        # pad to the same length
        min_len = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_len]
        target = target[..., :min_len]

        # 1 能量
        e_pred   = pred.pow(2)
        e_target = target.pow(2)

        # 2 Schroeder EDC（reverse cumsum）
        edc_pred   = torch.flip(torch.cumsum(torch.flip(e_pred, (-1,)),  dim=-1), (-1,))
        edc_target = torch.flip(torch.cumsum(torch.flip(e_target, (-1,)), dim=-1), (-1,))

        # 3 归一化
        edc_pred   = edc_pred   / (edc_pred[...,  :1] + self.eps)
        edc_target = edc_target / (edc_target[..., :1] + self.eps)

        if self.mode == "lin":
            # NMSE
            num = F.mse_loss(edc_pred, edc_target, reduction="none").sum(-1)
            den = edc_target.pow(2).sum(-1) + self.eps
            loss = num / den
        else:  # 'log'
            log_pred   = torch.log10(edc_pred   + self.eps)
            log_target = torch.log10(edc_target + self.eps)
            loss = (log_pred - log_target).abs().mean(-1)          # L1

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss        # shape [B]


# class WM_BCE_Loss(nn.Module):
#     def __init__(self, msg_len=16):
#         super(WM_BCE_Loss, self).__init__()
#         self.bce_logit = nn.BCEWithLogitsLoss()
    
#     def forward(self, msg_logit, msg):
#         loss = self.bce_logit(msg_logit, msg.float())
#         return loss

class WM_Hinge_Loss(nn.Module):
    def __init__(self, msg_len=16):
        super(WM_Hinge_Loss, self).__init__()
        self.margin = 1
    
    def forward(self, wm_logit, msg_logit, wm_exist,msg):
        msg, msg_logit = msg[wm_exist==1,:], msg_logit[wm_exist==1,:]
        msg = msg.float() * 2 - 1
        loss_msg = F.relu(1 - msg_logit * msg).mean()  
        wm_exist = wm_exist.float() * 2 - 1
        loss_wm = F.relu(1 - wm_logit * wm_exist).mean()
        loss = loss_msg + loss_wm
        return loss

if __name__ == "__main__":
    predicted = torch.randn(4, 1, 16555)
    target = torch.randn(4, 1, 16000)
    loss_msstft = MSSTFT_Loss()
    print(loss_msstft(predicted, target).item())

    loss_edc = EDCLoss()
    print(loss_edc(predicted, target).item())

    loss_early_snr = EarlySDR_Loss()
    print(loss_early_snr(predicted, target).item())

    predicted = torch.randn(4, 513, 100)
    target = torch.randn(4, 513, 99)
    loss_stft = STFT_Loss()
    print(loss_stft(predicted, target).item())

    msg_logit = torch.randn(4, 16)
    msg = torch.randint(0, 2, (4, 16), dtype=torch.float32)
    wm_logit = torch.randn(4, )
    wm_exist = torch.randint(0, 2, (4, ), dtype=torch.float32)
    # loss_wm = WM_BCE_Loss(msg_len=16)
    # print(loss_wm(msg_logit, msg).item())

    loss_wm_hinge = WM_Hinge_Loss(msg_len=16)
    print(loss_wm_hinge(wm_logit, msg_logit, wm_exist, msg).item())