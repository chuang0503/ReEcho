import torch
import torch.nn as nn
import torchaudio.transforms as T
# TODO: add pink noise and white noise, add pointed source noise
# add noise to audio
def add_noise(audio_batch, noise_batch, snr_db):
    """
    audio_batch: [B, 1, T]
    noise_batch: [B, 1, T]
    snr_db: [B]
    return: noisy_batch: [B, 1, T]
    """
    # [B,1]
    audio_power = torch.norm(audio_batch, p=2, dim=-1, keepdim=True)
    noise_power = torch.norm(noise_batch, p=2, dim=-1, keepdim=True)
    # snr_db: [B] -> [B, 1, 1] or float -> [1, 1, 1]
    if not torch.is_tensor(snr_db):
        snr_db = torch.tensor(snr_db, device=audio_batch.device)
    if snr_db.dim() == 0:
        snr_db = snr_db.view(1, 1, 1)
    else:
        snr_db = snr_db.view(-1, 1, 1)
    factor = (audio_power / (10 ** (snr_db / 20))) / (noise_power + 1e-10)  # [B, 1, 1]
    noisy_batch = audio_batch + factor * noise_batch
    return noisy_batch

class Distortion(nn.Module):
    def __init__(self, snr_db_range=(0, 20)):
        super(Distortion, self).__init__()
        self.snr_db_range = snr_db_range

    def forward(self, audio_batch, noise_batch):
        snr_db = torch.randint(self.snr_db_range[0], self.snr_db_range[1], (audio_batch.shape[0], 1), device=audio_batch.device)
        audio_noisy = add_noise(audio_batch, noise_batch, snr_db)
