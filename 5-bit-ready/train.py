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

# ─────────────────── logger ───────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
for h in (logging.StreamHandler(), logging.FileHandler("training.log")):
    h.setFormatter(fmt); logger.addHandler(h)

# ─────────────────── training loop ───────────────────
def train_loop(audio_dl_train, audio_dl_val, rir_dl_train, rir_dl_val, msg_len=8,
               epochs=100, lr=1e-4, device="cuda"):
    experiment_time = datetime.now().strftime('%m%d_%H%M')
    separator = ReEcho_Separator().to(device)
    generator = ReEcho_Generator().to(device)
    watermarker = ReEcho_WM(msg_len=msg_len).to(device)
    spec_transform = SpectrogramTransform().to(device)

    optimizer = torch.optim.Adam(
        [
            {"params": separator.parameters(), "lr": lr},
            {"params": generator.parameters(), "lr": lr},
            {"params": watermarker.parameters(), "lr": lr},
        ],
        lr=lr)
    
    stft_loss, ms_loss, wm_loss, edc_loss = STFT_Loss(), MSSTFT_Loss(), WM_Loss(msg_len=msg_len), EDCLoss()
    stft_loss.to(device)
    ms_loss.to(device)
    wm_loss.to(device)
    edc_loss.to(device)

    for ep in range(1, epochs + 1):
        # ---- train ----
        separator.train()
        generator.train()
        watermarker.train()

        # Training progress bar
        running_loss = 0.0
        running_dereverb_loss = 0.0
        running_rir_loss = 0.0
        running_decode_loss = 0.0
        train_pbar = tqdm(zip(audio_dl_train, rir_dl_train), total=len(audio_dl_train), desc=f"Epoch {ep}/{epochs} [Train]")
        for audio, rir in train_pbar:
            audio, rir = audio.to(device), rir.to(device)
            rs = torchaudio.functional.fftconvolve(audio, rir, mode='full')
            rs = rs[..., :audio.shape[-1]]

            # generate msg
            msg = torch.randint(0, 2, (audio.shape[0], watermarker.msg_len), device=device)

            optimizer.zero_grad()

            # feature extraction
            spec_masked, rir_emb = separator(rs)

            # watermark embedding 
            rir_emb_wm = watermarker.embedding(msg, rir_emb)

            # generation
            rir_est = generator(rir_emb_wm)

            # resynthesize and re-feature extraction
            audio_permuted = audio[torch.randperm(audio.shape[0])]
            rs_resyn = torchaudio.functional.fftconvolve(audio_permuted, rir_est, mode='full')
            rs_resyn = rs_resyn[:, :rs.shape[1]]

            # watermark extraction
            rs_resyn_spec = spec_transform(rs_resyn)
            msg_logit = watermarker.extraction(rs_resyn_spec, mode='logit')

            # loss
            dereverb_loss = stft_loss(spec_masked, spec_transform(audio))
            rir_loss = ms_loss(rir_est, rir) + edc_loss(rir_est, rir)
            decode_loss = wm_loss(msg_logit, msg)
            total_loss = dereverb_loss + rir_loss + decode_loss
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_dereverb_loss += dereverb_loss.item()
            running_rir_loss += rir_loss.item()
            running_decode_loss += decode_loss.item()
            
            # Update progress bar with current loss
            train_pbar.set_postfix({
                'total': f'{total_loss.item():.4f}',
                'dereverb': f'{dereverb_loss.item():.4f}',
                'rir': f'{rir_loss.item():.4f}',
                'decode': f'{decode_loss.item():.4f}'
            })
        tr_loss = running_loss / len(audio_dl_train)
        tr_dereverb_loss = running_dereverb_loss / len(audio_dl_train)
        tr_rir_loss = running_rir_loss / len(audio_dl_train)
        tr_decode_loss = running_decode_loss / len(audio_dl_train)
        
        print(f"Epoch {ep}/{epochs} | train_total: {tr_loss:.4f} | dereverb: {tr_dereverb_loss:.4f} | rir: {tr_rir_loss:.4f} | decode: {tr_decode_loss:.4f}", flush=True)
        logger.info(f"Epoch {ep}/{epochs} | train_total: {tr_loss:.4f} | dereverb: {tr_dereverb_loss:.4f} | rir: {tr_rir_loss:.4f} | decode: {tr_decode_loss:.4f}")

        # ---- val (every 5 epochs) ----
        if ep % 20 == 0:
            separator.eval()
            generator.eval()
            watermarker.eval()
            
            val_dereverb_loss = 0.0
            val_rir_loss = 0.0
            val_ber = 0.0
            # Validation progress bar
            with torch.no_grad():
                val_pbar = tqdm(zip(audio_dl_val, rir_dl_val), total=len(audio_dl_val), desc=f"Epoch {ep}/{epochs} [Val]")
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
                    rs_resyn = rs_resyn[:, :rs.shape[1]]

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
            
            print(f"Epoch {ep}/{epochs} | train_total: {tr_loss:.4f} | val_dereverb: {va_dereverb_loss:.4f} | val_rir: {va_rir_loss:.4f} | val_ber: {va_ber:.4f}", flush=True)
            logger.info(f"Epoch {ep}/{epochs} | train_total: {tr_loss:.4f} | val_dereverb: {va_dereverb_loss:.4f} | val_rir: {va_rir_loss:.4f} | val_ber: {va_ber:.4f}")      
            
            checkpoint_dir = f"checkpoints/{experiment_time}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            torch.save({
                'epoch': ep,
                'separator_state_dict': separator.state_dict(),
                'generator_state_dict': generator.state_dict(),
                'watermarker_state_dict': watermarker.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': tr_loss,
                'train_dereverb_loss': tr_dereverb_loss,
                'train_rir_loss': tr_rir_loss,
                'train_decode_loss': tr_decode_loss,
            }, f'{checkpoint_dir}/rir_model_epoch_{ep}.pth')
            print(f"Model saved at epoch {ep}")
            logger.info(f"Model saved at epoch {ep}")



# ─────────────────── main ───────────────────
def main():
    audio_dataset_train = MyLibriSpeech(url="dev-clean",sr=16000, duration=2)
    audio_dataset_val = MyLibriSpeech(url="dev-clean",sr=16000, duration=2)
    rir_dataset_full = RIRS_Dataset(sr=16000, duration=2)

    # split rir dataset
    total_len = len(rir_dataset_full)
    train_size = int(0.8 * total_len)
    val_size = total_len - train_size
    rir_dataset_train, rir_dataset_val = random_split(
        rir_dataset_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    audio_dl_train, rir_dl_train = get_dataloader(audio_dataset_train, rir_dataset_train, batch_size=40, num_workers=64, persistent_workers=True, pin_memory=True)
    audio_dl_val, rir_dl_val = get_dataloader(audio_dataset_val, rir_dataset_val, batch_size=40, num_workers=64, persistent_workers=True, pin_memory=True)
    print("Starting training...")
    # just for test
    train_loop(audio_dl_train, audio_dl_val, rir_dl_train, rir_dl_val, msg_len=5, epochs=500, lr=1e-4)

if __name__ == "__main__":
    main()
