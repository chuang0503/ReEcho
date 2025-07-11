# train_conformer_mask.py
from sympy.polys.numberfields.minpoly import rs_compose_add
import sys, logging, torch, torchaudio
sys.path.append("/home/c3-server1/Documents/Chenpei/re-rir")
import os
from datetime import datetime

from model import *
from wm_scheduler import TauTeacherScheduler
from data.util        import unpickle_data
from dataloader       import RIR_Dataset, create_dataloader
from loss_fn          import MSSTFT_Loss, STFT_Loss, WM_Loss
from icecream import ic; ic.disable()

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

# ─────────────────── logger ───────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
for h in (logging.StreamHandler(), logging.FileHandler("training.log")):
    h.setFormatter(fmt); logger.addHandler(h)

# ─────────────────── training loop ───────────────────
def train_loop(train_dl, val_dl, msg_len=8,
               epochs=100, lr=1e-4, device="cuda"):
    experiment_time = datetime.now().strftime('%m%d_%H%M')
    separator = ReEcho_Separator().to(device)
    generator = ReEcho_Generator().to(device)
    watermarker = ReEcho_WM(msg_len=msg_len).to(device)
    spec_transform = SpectrogramTransform().to(device)

    wm_scheduler = TauTeacherScheduler(total_steps=epochs*len(train_dl))
    optimizer = torch.optim.Adam(
        [
            {"params": separator.parameters(), "lr": lr},
            {"params": generator.parameters(), "lr": lr},
            {"params": watermarker.parameters(), "lr": lr},
        ],
        lr=lr)
    
    stft_loss, ms_loss, wm_loss = STFT_Loss(), MSSTFT_Loss(), WM_Loss(msg_len=msg_len)
    stft_loss.to(device)
    ms_loss.to(device)
    wm_loss.to(device)
    for ep in range(1, epochs + 1):
        # ---- train ----
        separator.train()
        generator.train()
        watermarker.train()

        # Use tqdm with minimal output for nohup
        train_pbar = tqdm(train_dl, desc=f"Epoch {ep}/{epochs} [Train]", 
                         leave=False, disable=True)  # disable=True reduces output
        running_loss = 0.0
        running_dereverb_loss = 0.0
        running_rir_loss = 0.0
        running_decode_loss = 0.0
        for rs, clean, rir in train_pbar:
            rs, clean, rir = rs.to(device), clean.to(device), rir.to(device)
            # generate msg
            msg = torch.randint(0, 2, (rs.shape[0], watermarker.msg_len), device=device)

            optimizer.zero_grad()

            # feature extraction
            spec_masked, rir_emb = separator(rs)

            # watermark embedding 
            rir_emb_wm = watermarker.embedding(msg, rir_emb)

            # generation
            rir_est = generator(rir_emb_wm)

            # resynthesize and re-feature extraction
            clean_permuted = clean[torch.randperm(clean.shape[0])]
            rs_resyn = torchaudio.functional.fftconvolve(clean_permuted, rir_est, mode='full')
            rs_resyn = rs_resyn[:, :rs.shape[1]]

            # watermark extraction
            rs_resyn_spec = spec_transform(rs_resyn)
            msg_logit = watermarker.extraction(rs_resyn_spec, mode='logit')

            # loss
            dereverb_loss = stft_loss(spec_masked, spec_transform(clean))
            rir_loss = ms_loss(rir_est, rir)
            decode_loss = wm_loss(msg_logit, msg)
            total_loss = dereverb_loss + rir_loss + decode_loss
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_dereverb_loss += dereverb_loss.item()
            running_rir_loss += rir_loss.item()
            running_decode_loss += decode_loss.item()
        tr_loss = running_loss / len(train_dl)
        tr_dereverb_loss = running_dereverb_loss / len(train_dl)
        tr_rir_loss = running_rir_loss / len(train_dl)
        tr_decode_loss = running_decode_loss / len(train_dl)
        
        print(f"Epoch {ep}/{epochs} | train_total: {tr_loss:.4f} | dereverb: {tr_dereverb_loss:.4f} | rir: {tr_rir_loss:.4f} | decode: {tr_decode_loss:.4f}", flush=True)
        logger.info(f"Epoch {ep}/{epochs} | train_total: {tr_loss:.4f} | dereverb: {tr_dereverb_loss:.4f} | rir: {tr_rir_loss:.4f} | decode: {tr_decode_loss:.4f}")

        # ---- val (every 5 epochs) ----
        if ep % 5 == 0:
            separator.eval()
            generator.eval()
            watermarker.eval()
            
            val_dereverb_loss = 0.0
            val_rir_loss = 0.0
            val_ber = 0.0
            # Use tqdm with minimal output for nohup
            val_pbar = tqdm(val_dl, desc=f"Epoch {ep}/{epochs} [Val]", 
                           leave=False, disable=True)  # disable=True reduces output
            
            with torch.no_grad():
                for rs, clean, rir in val_pbar:
                    rs, clean, rir = rs.to(device), clean.to(device), rir.to(device)
                    msg = torch.randint(0, 2, (rs.shape[0], watermarker.msg_len), device=device)

                    spec_masked, rir_emb = separator(rs)
                    rir_emb_wm = watermarker.embedding(msg, rir_emb)
                    rir_est = generator(rir_emb_wm)

                    # resynthesize and re-feature extraction
                    clean_permuted = clean[torch.randperm(clean.shape[0])]
                    rs_resyn = torchaudio.functional.fftconvolve(clean_permuted, rir_est, mode='full')
                    rs_resyn = rs_resyn[:, :rs.shape[1]]

                    # watermark extraction
                    rs_resyn_spec = spec_transform(rs_resyn)
                    msg_logit = watermarker.extraction(rs_resyn_spec, mode='logit')

                    # loss
                    dereverb_loss = stft_loss(spec_masked, spec_transform(clean))
                    rir_loss = ms_loss(rir_est, rir)
                    ber = (msg_logit != msg).float().mean()
                    
                    val_dereverb_loss += dereverb_loss.item()
                    val_rir_loss += rir_loss.item()
                    val_ber += ber.item()
            va_dereverb_loss = val_dereverb_loss / len(val_dl)
            va_rir_loss = val_rir_loss / len(val_dl)
            va_ber = val_ber / len(val_dl)
            
            print(f"Epoch {ep}/{epochs} | train_total: {tr_loss:.4f} | val_dereverb: {va_dereverb_loss:.4f} | val_rir: {va_rir_loss:.4f} | val_ber: {va_ber:.4f}", flush=True)
            logger.info(f"Epoch {ep}/{epochs} | train_total: {tr_loss:.4f} | val_dereverb: {va_dereverb_loss:.4f} | val_rir: {va_rir_loss:.4f} | val_ber: {va_ber:.4f}")
        else:
            print(f"Epoch {ep}/{epochs} | train_total: {tr_loss:.4f} | val --", flush=True)
            logger.info(f"Epoch {ep}/{epochs} | train_total: {tr_loss:.4f} | val --")
        
        # Save model every 20 epochs
        if ep % 20 == 0:
            # Create checkpoints directory if it doesn't exist
            
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
                'val_dereverb_loss': va_dereverb_loss if ep % 5 == 0 else None,
                'val_rir_loss': va_rir_loss if ep % 5 == 0 else None,
                'val_ber': va_ber if ep % 5 == 0 else None,
            }, f'{checkpoint_dir}/rir_model_epoch_{ep}.pth')
            print(f"Model saved at epoch {ep}")
            logger.info(f"Model saved at epoch {ep}")



# ─────────────────── main ───────────────────
def main():
    print("Loading training data...")
    # train_entries = unpickle_data("../data/synthetic/dev-real-rir_real_100.pkl")
    train_entries = unpickle_data("../data/synthetic/test-real-rir_real_2.pkl")
    print(f"Loaded {len(train_entries)} training entries")
    
    print("Loading validation data...")
    val_entries   = unpickle_data("../data/synthetic/test-real-rir_real_2.pkl")
    print(f"Loaded {len(val_entries)} validation entries")

    print("Creating datasets...")
    tr_ds = RIR_Dataset(train_entries)
    va_ds = RIR_Dataset(val_entries)
    print(f"Training dataset size: {len(tr_ds)}")
    print(f"Validation dataset size: {len(va_ds)}")
    
    print("Creating dataloaders...")
    tr_dl, va_dl = create_dataloader(tr_ds, va_ds, batch_size=32, num_workers=16)
    print(f"Training batches: {len(tr_dl)}")
    print(f"Validation batches: {len(va_dl)}")
    
    print("Starting training...")
    train_loop(tr_dl, va_dl, msg_len=16, epochs=100, lr=1e-4)

if __name__ == "__main__":
    main()
