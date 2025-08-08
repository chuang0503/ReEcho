"""Distributed training script for ReEcho model."""

import sys
import logging
import os
from datetime import datetime
import warnings

import torch
import torchaudio
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import random_split, ConcatDataset

from model import *
from new_dataloader import RIRS_Dataset, get_dataloader_ddp, MyLibriSpeech, BUT_Dataset, add_noise, Noise_Dataset
from loss_fn import MSSTFT_Loss, STFT_Loss, WM_Hinge_Loss, EDCLoss
from icecream import ic

ic.disable()
warnings.filterwarnings("ignore")

# ddp
def ddp_setup():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

# ─────────────────── logger ───────────────────
local_rank = ddp_setup() 

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
if local_rank == 0:
    for h in (logging.StreamHandler(), logging.FileHandler("training.log")):
        h.setFormatter(fmt)
        logger.addHandler(h)

# ─────────────────── training loop ───────────────────
def train_loop(audio_dl_train, rir_dl_train, audio_dl_val, rir_dl_val, noise_dl_train, noise_dl_val, local_rank, msg_len=5,
               epochs=200, lr=1e-4):
    device = f"cuda:{local_rank}"
    experiment_time = datetime.now().strftime('%m%d_%H%M')
    separator = ReEcho_Separator().to(device)
    generator = ReEcho_Generator().to(device)
    watermarker = ReEcho_WM(msg_len=msg_len).to(device)
    spec_transform = SpectrogramTransform().to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": separator.parameters(), "lr": lr},
            {"params": generator.parameters(), "lr": lr},
            {"params": watermarker.parameters(), "lr": lr},
        ],
        lr=lr)

    # # ------------ load checkpoint ------------
    # checkpoint = torch.load(f"save/trained_WM_detector.pth", map_location=device) #4.41
    # separator.load_state_dict(checkpoint['separator_state_dict'])
    # generator.load_state_dict(checkpoint['generator_state_dict'])
    # watermarker.load_state_dict(checkpoint['watermarker_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # ------------ ddp ------------
    separator = DDP(separator, device_ids=[local_rank])
    generator = DDP(generator, device_ids=[local_rank])
    watermarker = DDP(watermarker, device_ids=[local_rank])
    
    
    stft_loss, ms_loss, wm_loss, edc_loss = STFT_Loss(), MSSTFT_Loss(), WM_Hinge_Loss(msg_len=msg_len), EDCLoss()
    stft_loss.to(device)
    ms_loss.to(device)
    wm_loss.to(device)
    edc_loss.to(device)
    for ep in range(1, epochs + 1):
        # ---- train ----
        separator.train()
        generator.train()
        watermarker.train()
        audio_dl_train.sampler.set_epoch(ep)

        # Use tqdm with minimal output for nohup
        train_pbar = tqdm(zip(audio_dl_train, rir_dl_train, noise_dl_train), desc=f"Epoch {ep}/{epochs} [Train]", 
                         leave=False, disable=True)
        running_loss = 0.0
        running_dereverb_loss = 0.0
        running_rir_loss = 0.0
        running_decode_loss = 0.0
        for audio, rir, noise in train_pbar:
            audio, rir, noise = audio.to(device), rir.to(device), noise.to(device)
            rs = torchaudio.functional.fftconvolve(audio, rir, mode='full')
            rs = rs[..., :audio.shape[-1]]
            

            # add noise # this is usuaally safe in training
            snr_db = torch.randint(0, 20, (audio.shape[0], 1), device=device)
            rs = add_noise(rs, noise, snr_db) if torch.rand(1) < 0.5 else rs

            # generate msg
            msg = torch.randint(0, 2, (rs.shape[0], msg_len), device=device, dtype=torch.long)
            # update wm_scheduler
            optimizer.zero_grad()

            # feature extraction
            spec_masked, rir_emb = separator(rs)
            rir_emb_wm = watermarker.module.embedding(msg, rir_emb)
            rir_est = generator(rir_emb_wm) 
            # resynthesize
            audio_permuted = audio[torch.randperm(audio.shape[0])]
            rs_resyn = torchaudio.functional.fftconvolve(audio_permuted, rir_est, mode='full')
            rs_resyn = rs_resyn[..., :rs.shape[-1]]
            

            # add noise (test) # comment out if it affects convergence. Turn on when training on checkpoint.
            snr_db = torch.randint(0, 20, (audio.shape[0], 1), device=device)
            rs_resyn = add_noise(rs_resyn, noise, snr_db) if torch.rand(1) < 0.5 else rs_resyn

            # re-feature extraction
            wm_false = torch.zeros(rs.shape[0], device=device, dtype=torch.long)
            wm_true = torch.ones(rs_resyn.shape[0], device=device, dtype=torch.long)
            wm_exist = torch.cat([wm_false, wm_true], dim=0)
            rs_detect = torch.cat([rs, rs_resyn], dim=0)
            msg = torch.cat([msg, msg], dim=0)
            idx = torch.randperm(rs_detect.shape[0])
            rs_detect, wm_exist, msg = rs_detect[idx], wm_exist[idx], msg[idx]
            #
            rs_detect_spec = spec_transform(rs_detect)
            wm_logit, msg_logit = watermarker.module.extraction(rs_detect_spec, mode='train')
            # loss
            dereverb_loss = stft_loss(spec_masked, spec_transform(audio))
            rir_loss = ms_loss(rir_est, rir) + edc_loss(rir_est, rir)
            decode_loss = wm_loss(wm_logit, msg_logit, wm_exist, msg) 
            total_loss = dereverb_loss + rir_loss + decode_loss
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_dereverb_loss += dereverb_loss.item()
            running_rir_loss += rir_loss.item()
            running_decode_loss += decode_loss.item()

        tr_loss = running_loss / len(audio_dl_train)
        tr_dereverb_loss = running_dereverb_loss / len(audio_dl_train)
        tr_rir_loss = running_rir_loss / len(audio_dl_train)
        tr_decode_loss = running_decode_loss / len(audio_dl_train)
        if local_rank == 0:
            logger.info(f"Epoch {ep}/{epochs} | train_total: {tr_loss:.4f} | dereverb: {tr_dereverb_loss:.4f} | rir: {tr_rir_loss:.4f} | decode: {tr_decode_loss:.4f}")

        # ---- val (every 5 epochs) ----
        if ep % 5 == 0:
            separator.eval()
            generator.eval()
            watermarker.eval()

            val_dereverb_loss = 0.0
            val_rir_loss = 0.0
            val_wm_acc = 0.0
            val_ber = 0.0
            
            # Use tqdm with minimal output for nohup
            val_pbar = tqdm(zip(audio_dl_val, rir_dl_val, noise_dl_val), desc=f"Epoch {ep}/{epochs} [Val]", 
                           leave=False, disable=True)
            
            with torch.no_grad():
                for audio, rir, noise in val_pbar:
                    audio, rir, noise = audio.to(device), rir.to(device), noise.to(device)
                    rs = torchaudio.functional.fftconvolve(audio, rir, mode='full')
                    rs = rs[..., :audio.shape[-1]]
                    

                    # add noise
                    snr_db = torch.randint(0, 20, (audio.shape[0], 1), device=device)
                    rs = add_noise(rs, noise, snr_db) if torch.rand(1) < 0.5 else rs

                    # generate msg
                    msg = torch.randint(0, 2, (rs.shape[0], msg_len), device=device, dtype=torch.long)
                    # feature extraction
                    spec_masked, rir_emb = separator(rs)
                    rir_emb_wm = watermarker.module.embedding(msg, rir_emb)
                    rir_est = generator(rir_emb_wm)
                    # resynthesize
                    audio_permuted = audio[torch.randperm(audio.shape[0])]
                    rs_resyn = torchaudio.functional.fftconvolve(audio_permuted, rir_est, mode='full')
                    rs_resyn = rs_resyn[..., :rs.shape[-1]]
                    

                    # add noise (test) # comment out if it affects convergence. Turn on when training on checkpoint.
                    snr_db = torch.randint(0, 20, (audio.shape[0], 1), device=device)
                    rs_resyn = add_noise(rs_resyn, noise, snr_db) if torch.rand(1) < 0.5 else rs_resyn

                    # re-feature extraction
                    wm_false = torch.zeros(rs.shape[0], device=device, dtype=torch.long)
                    wm_true = torch.ones(rs_resyn.shape[0], device=device, dtype=torch.long)
                    wm_exist = torch.cat([wm_false, wm_true], dim=0) # [2B]
                    rs_detect = torch.cat([rs, rs_resyn], dim=0)
                    msg = torch.cat([msg, msg], dim=0)
                    idx = torch.randperm(rs_detect.shape[0])
                    rs_detect, wm_exist, msg = rs_detect[idx], wm_exist[idx], msg[idx]
                    #
                    rs_detect_spec = spec_transform(rs_detect)
                    wm_logit, msg_logit = watermarker.module.extraction(rs_detect_spec, mode='test')
                    # loss
                    dereverb_loss = stft_loss(spec_masked, spec_transform(audio))
                    rir_loss = ms_loss(rir_est, rir) + edc_loss(rir_est, rir)
                    wm_acc = (wm_logit == wm_exist).float().mean()
                    ber = (msg_logit[wm_exist==1,:] != msg[wm_exist==1,:]).float().mean()

                    val_dereverb_loss += dereverb_loss.item()
                    val_rir_loss += rir_loss.item()
                    val_wm_acc += wm_acc.item()
                    val_ber += ber.item()
            va_dereverb_loss = val_dereverb_loss / len(audio_dl_val)
            va_rir_loss = val_rir_loss / len(audio_dl_val)
            va_wm_acc = val_wm_acc / len(audio_dl_val)
            va_ber = val_ber / len(audio_dl_val)
            if local_rank == 0:
                print(f"Epoch {ep}/{epochs} | train_total: {tr_loss:.4f} | val_dereverb: {va_dereverb_loss:.4f} | val_rir: {va_rir_loss:.4f} | val_wm_acc: {va_wm_acc:.4f} | val_ber: {va_ber:.4f}", flush=True)
                logger.info(f"Epoch {ep}/{epochs} | train_total: {tr_loss:.4f} | val_dereverb: {va_dereverb_loss:.4f} | val_rir: {va_rir_loss:.4f} | val_wm_acc: {va_wm_acc:.4f} | val_ber: {va_ber:.4f}")
        else:
            if local_rank == 0:
                print(f"Epoch {ep}/{epochs} | train_total: {tr_loss:.4f} | val --", flush=True)
                logger.info(f"Epoch {ep}/{epochs} | train_total: {tr_loss:.4f} | val --")
        
        # Save model every 10 epochs
        if ep % 10 == 0 and local_rank == 0:
            # Create checkpoints directory if it doesn't exist
            
            checkpoint_dir = f"checkpoints/{experiment_time}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            torch.save({
                'epoch': ep,
                'separator_state_dict': separator.module.state_dict(),
                'generator_state_dict': generator.module.state_dict(),
                'watermarker_state_dict': watermarker.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': tr_loss,
                'train_dereverb_loss': tr_dereverb_loss,
                'train_rir_loss': tr_rir_loss,
                'val_dereverb_loss': va_dereverb_loss if ep % 5 == 0 else None,
                'val_rir_loss': va_rir_loss if ep % 5 == 0 else None,
                'val_wm_acc': va_wm_acc if ep % 5 == 0 else None,
                'val_ber': va_ber if ep % 5 == 0 else None,
            }, f'{checkpoint_dir}/rir_model_epoch_{ep}.pth')
            logger.info(f"Model saved at epoch {ep}")



# ─────────────────── main ───────────────────
def main():
    print("Loading training data...")
    audio_dataset_train = MyLibriSpeech(url="train-clean-100",sr=16000, duration=2)
    audio_dataset_val = MyLibriSpeech(url="dev-clean",sr=16000, duration=2)
    rir_1 = BUT_Dataset(sr=16000, duration=2)
    rir_2 = RIRS_Dataset(sr=16000, duration=2)

    rir_dataset = ConcatDataset([rir_1, rir_2])
    rir_train, rir_val = random_split(rir_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

    noise_dataset = Noise_Dataset(sr=16000, duration=2)

    audio_dl_train, rir_dl_train, noise_dl_train = get_dataloader_ddp(audio_dataset_train, rir_train, noise_dataset, batch_size=40, num_workers=64, persistent_workers=True, pin_memory=True)
    audio_dl_val, rir_dl_val, noise_dl_val = get_dataloader_ddp(audio_dataset_val, rir_val, noise_dataset, batch_size=40, num_workers=64, persistent_workers=True, pin_memory=True)
    
    print("Starting training...")
    train_loop(audio_dl_train, rir_dl_train, audio_dl_val, rir_dl_val, noise_dl_train, noise_dl_val, local_rank, msg_len=5, epochs=500, lr=1e-4)

if __name__ == "__main__":
    main()
