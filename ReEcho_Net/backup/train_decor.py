# train_conformer_mask.py
import sys, logging, torch, torchaudio
sys.path.append("/home/c3-server1/Documents/Chenpei/re-rir")
import os
from datetime import datetime

from model_fins_decor import *
from data.util        import unpickle_data
from dataloader       import RIR_Dataset, create_dataloader
from loss_fn          import MSSTFT_Loss
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
def train_loop(train_dl, val_dl,
               epochs=100, lr=1e-4, device="cuda"):
    experiment_time = datetime.now().strftime('%m%d_%H%M')
    spec_tf = SpectrogramTransform()
    mel_tf  = MelScaleTransform()
    enc     = ConformerEncoder()
    mask_net = MaskNet_IRM()
    rir_emb_net = RIREmbedding()
    rir_dec_net = RIRDecoder_Decor()

    spec_tf.to(device)
    mel_tf.to(device)
    enc.to(device)
    mask_net.to(device)
    rir_emb_net.to(device)
    rir_dec_net.to(device)

    optimizer = torch.optim.Adam(
        [
            {"params": enc.parameters(), "lr": lr},
            {"params": mask_net.parameters(), "lr": lr},
            {"params": rir_emb_net.parameters(), "lr": lr},
            {"params": rir_dec_net.parameters(), "lr": lr},
        ],
        lr=lr)
    
    loss_fn = MSSTFT_Loss()
    loss_fn.to(device)
    
    for ep in range(1, epochs + 1):
        # ---- train ----
        enc.train()
        mask_net.train()
        rir_emb_net.train()
        rir_dec_net.train()

        # Use tqdm with minimal output for nohup
        train_pbar = tqdm(train_dl, desc=f"Epoch {ep}/{epochs} [Train]", 
                         leave=False, disable=True)  # disable=True reduces output
        running_loss = 0.0
        running_dereverb_loss = 0.0
        running_rir_loss = 0.0
        
        for rs, clean, rir in train_pbar:
            rs, clean, rir = rs.to(device), clean.to(device), rir.to(device)
            clean_spec = spec_tf(clean)
            reverb_spec = spec_tf(rs)

            optimizer.zero_grad()

            # feature extraction
            feat = enc(reverb_spec)
            # dereverb mask
            mask  = mask_net(feat) # option (a) mel input (b) linear input
            dereverb_spec = mask * reverb_spec

            # rir generation
            rir_emb, seed = rir_emb_net(feat)
            rir_dec,_,_ = rir_dec_net(rir_emb, seed)

            rir_len = rir_dec.shape[-1]
            rir_gt = rir[..., :rir_len]

            # loss
            dereverb_loss = torch.nn.functional.l1_loss(dereverb_spec, clean_spec) * 100
            rir_loss = loss_fn(rir_dec, rir_gt)
            total_loss = dereverb_loss + rir_loss
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_dereverb_loss += dereverb_loss.item()
            running_rir_loss += rir_loss.item()

        tr_loss = running_loss / len(train_dl)
        tr_dereverb_loss = running_dereverb_loss / len(train_dl)
        tr_rir_loss = running_rir_loss / len(train_dl)
        
        print(f"Epoch {ep}/{epochs} | train_total: {tr_loss:.4f} | dereverb: {tr_dereverb_loss:.4f} | rir: {tr_rir_loss:.4f}", flush=True)
        logger.info(f"Epoch {ep}/{epochs} | train_total: {tr_loss:.4f} | dereverb: {tr_dereverb_loss:.4f} | rir: {tr_rir_loss:.4f}")

        # ---- val (every 5 epochs) ----
        if ep % 5 == 0:
            enc.eval()
            mask_net.eval()
            rir_emb_net.eval()
            rir_dec_net.eval()
            
            val_loss = 0.0
            val_dereverb_loss = 0.0
            val_rir_loss = 0.0
            
            # Use tqdm with minimal output for nohup
            val_pbar = tqdm(val_dl, desc=f"Epoch {ep}/{epochs} [Val]", 
                           leave=False, disable=True)  # disable=True reduces output
            
            with torch.no_grad():
                for rs, clean, rir in val_pbar:
                    rs, clean, rir = rs.to(device), clean.to(device), rir.to(device)
                    clean_spec = spec_tf(clean)
                    clean_mel = mel_tf(clean_spec)
                    reverb_spec = spec_tf(rs)
                    reverb_mel = mel_tf(reverb_spec)

                    feat = enc(reverb_spec)
                    mask = mask_net(feat)
                    dereverb_spec = mask * reverb_spec

                    rir_emb, seed = rir_emb_net(feat)
                    rir_dec,_,_ = rir_dec_net(rir_emb, seed)

                    rir_len = rir_dec.shape[-1]
                    rir_gt = rir[..., :rir_len]

                    dereverb_loss = torch.nn.functional.l1_loss(dereverb_spec, clean_spec) * 100
                    rir_loss = loss_fn(rir_dec, rir_gt)
                    total_loss = dereverb_loss + rir_loss
                    
                    val_loss += total_loss.item()
                    val_dereverb_loss += dereverb_loss.item()
                    val_rir_loss += rir_loss.item()
            
            va_loss = val_loss / len(val_dl)
            va_dereverb_loss = val_dereverb_loss / len(val_dl)
            va_rir_loss = val_rir_loss / len(val_dl)
            
            print(f"Epoch {ep}/{epochs} | train_total: {tr_loss:.4f} | val_total: {va_loss:.4f} | val_dereverb: {va_dereverb_loss:.4f} | val_rir: {va_rir_loss:.4f}", flush=True)
            logger.info(f"Epoch {ep}/{epochs} | train_total: {tr_loss:.4f} | val_total: {va_loss:.4f} | val_dereverb: {va_dereverb_loss:.4f} | val_rir: {va_rir_loss:.4f}")
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
                'encoder_state_dict': enc.state_dict(),
                'mask_net_state_dict': mask_net.state_dict(),
                'rir_emb_net_state_dict': rir_emb_net.state_dict(),
                'rir_dec_net_state_dict': rir_dec_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': tr_loss,
                'train_dereverb_loss': tr_dereverb_loss,
                'train_rir_loss': tr_rir_loss,
                'val_loss': va_loss if ep % 5 == 0 else None,
                'val_dereverb_loss': va_dereverb_loss if ep % 5 == 0 else None,
                'val_rir_loss': va_rir_loss if ep % 5 == 0 else None,
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
    tr_dl, va_dl = create_dataloader(tr_ds, va_ds, batch_size=64, num_workers=16)
    print(f"Training batches: {len(tr_dl)}")
    print(f"Validation batches: {len(va_dl)}")
    
    print("Starting training...")
    train_loop(tr_dl, va_dl, epochs=100, lr=1e-4)

if __name__ == "__main__":
    main()
