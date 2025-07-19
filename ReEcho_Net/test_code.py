import sys
import logging
import torch
import torchaudio
import os
from datetime import datetime
import warnings

from torch.utils.data import random_split

from model import *
from new_dataloader import get_dataloader, MyLibriSpeech, RIRS_Dataset
from loss_fn import MSSTFT_Loss, STFT_Loss, WM_Loss, EDCLoss
from icecream import ic
from tqdm import tqdm

ic.disable()
warnings.filterwarnings("ignore")

# ─────────────────── training loop ───────────────────
def test_loop(audio_dl_val, rir_dl_val, msg_len=8, device="cuda"):
    separator = ReEcho_Separator().to(device)
    generator = ReEcho_Generator().to(device)
    watermarker = ReEcho_WM(msg_len=msg_len).to(device)
    spec_transform = SpectrogramTransform().to(device)
    
    stft_loss, ms_loss, wm_loss, edc_loss = STFT_Loss(), MSSTFT_Loss(), WM_Loss(msg_len=msg_len), EDCLoss()
    stft_loss.to(device)
    ms_loss.to(device)
    wm_loss.to(device)
    edc_loss.to(device)

    ckpt = torch.load("checkpoints/0719_0145/rir_model_epoch_200.pth", map_location=device) 
    separator.load_state_dict(ckpt['separator_state_dict'])
    generator.load_state_dict(ckpt['generator_state_dict'])
    watermarker.load_state_dict(ckpt['watermarker_state_dict'])

    if True:
        separator.eval()
        generator.eval()
        watermarker.eval()
        
        val_dereverb_loss = 0.0
        val_rir_loss = 0.0
        val_ber = 0.0
        # Validation progress bar
        with torch.no_grad():
            val_pbar = tqdm(zip(audio_dl_val, rir_dl_val), total=len(audio_dl_val), desc=f"Epoch [Val]")
            for audio, rir in val_pbar:
                audio, rir = audio.to(device), rir.to(device)
                rs = torchaudio.functional.fftconvolve(audio, rir, mode='full')
                rs = rs[..., :audio.shape[-1]]
                msg = torch.randint(0, 2, (audio.shape[0], watermarker.msg_len), device=device)

                spec_masked, rir_emb = separator(rs)
                rir_emb_wm = watermarker.embedding(msg, rir_emb)
                rir_est = generator(rir_emb_wm)

                # resynthesize and re-feature extraction
                audio_permuted = audio[torch.randperm(audio.shape[0])]
                rs_resyn = torchaudio.functional.fftconvolve(audio_permuted, rir_est, mode='full')
                rs_resyn = rs_resyn[..., :rs.shape[-1]]

                # watermark extraction
                rs_resyn_spec = spec_transform(rs_resyn)
                msg_logit = watermarker.extraction(rs_resyn_spec, mode='msg')

                # loss
                dereverb_loss = stft_loss(spec_masked, spec_transform(audio))
                rir_loss = ms_loss(rir_est, rir)
                ber = (msg_logit != msg).float().mean()
                
                val_dereverb_loss += dereverb_loss.item()
                val_rir_loss += rir_loss.item()
                val_ber += ber.item()
                
                # Update validation progress bar
                val_pbar.set_postfix({
                    'dereverb': f'{dereverb_loss.item():.4f}',
                    'rir': f'{rir_loss.item():.4f}',
                    'ber': f'{ber.item():.4f}'
                })
            va_dereverb_loss = val_dereverb_loss / len(audio_dl_val)
            va_rir_loss = val_rir_loss / len(audio_dl_val)
            va_ber = val_ber / len(audio_dl_val)
        
            print(f"val_dereverb: {va_dereverb_loss:.4f} | val_rir: {va_rir_loss:.4f} | val_ber: {va_ber:.4f}", flush=True)

            return audio, rir, rs, spec_masked, rir_est



# ─────────────────── main ───────────────────
def main():
    audio_dataset_val = MyLibriSpeech(url="dev-clean",sr=16000, duration=2)
    rir_dataset_full = RIRS_Dataset(sr=16000, duration=2)

    audio_dl_val, rir_dl_val = get_dataloader(audio_dataset_val, rir_dataset_full, batch_size=40, num_workers=64, persistent_workers=True, pin_memory=True)
    print("Starting testing...")
    # just for test
    test_loop(audio_dl_val, rir_dl_val, msg_len=5)

if __name__ == "__main__":
    main()
