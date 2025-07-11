import os
import sys
import logging
sys.path.append('/home/c3-server1/Documents/Chenpei/re-rir')

import pickle
import torch
import torchaudio
import numpy as np
from data.util import unpickle_data
from torchaudio.functional import resample
import soundfile as sf
from torch.utils.data.distributed import DistributedSampler


from icecream import ic
ic.disable()

# Set up logging
logger = logging.getLogger(__name__)

class RIR_Dataset(torch.utils.data.Dataset):
    def __init__(self, rir_entries, duration=2, sr=16000): # TODO: rs_synthesize clip rir duration to 2s
        self.rir_entries = self.filter_valid_entries(rir_entries)
        self.duration = duration
        self.sr = sr

    def filter_valid_entries(self, rir_entries):
        """Filter out entries with invalid audio files"""
        valid_entries = []
        for i, entry in enumerate(rir_entries):
            try:
                # Test if all three audio files can be loaded
                self.load_audio_robust(entry['rs_path'])
                self.load_audio_robust(entry['speech_path'])
                self.load_audio_robust(entry['rir_path'])
                valid_entries.append(entry)
            except Exception as e:
                logger.warning(f"Skipping entry {i} due to invalid audio files: {e}")
        logger.info(f"Filtered {len(rir_entries)} entries to {len(valid_entries)} valid entries")
        return valid_entries

    def __len__(self):
        return len(self.rir_entries)
    
    def load_audio_robust(self, file_path):
        """Load audio file with fallback to soundfile if torchaudio fails"""
        try:
            # Try torchaudio first
            audio, sr = torchaudio.load(file_path)
            return audio, sr
        except Exception as e:
            ic(f"torchaudio failed to load {file_path}: {e}")
            try:
                # Fallback to soundfile
                audio, sr = sf.read(file_path)
                # Convert to torch tensor and ensure correct shape
                if len(audio.shape) == 1:
                    audio = audio.reshape(1, -1)
                else:
                    audio = audio.T  # soundfile returns (samples, channels), we want (channels, samples)
                audio = torch.from_numpy(audio).float()
                return audio, sr
            except Exception as e2:
                ic(f"soundfile also failed to load {file_path}: {e2}")
                raise
    
    def __getitem__(self, idx):
        rir_entry = self.rir_entries[idx]
        # Load audio files with specified sample rate and length
        rs_audio, sr_rs = self.load_audio_robust(rir_entry['rs_path'])
        if sr_rs != self.sr:
            rs_audio = resample(rs_audio, orig_freq=sr_rs, new_freq=self.sr)

        speech_audio, sr_speech = self.load_audio_robust(rir_entry['speech_path'])
        if sr_speech != self.sr:
            speech_audio = resample(speech_audio, orig_freq=sr_speech, new_freq=self.sr)

        rir_audio, sr_rir = self.load_audio_robust(rir_entry['rir_path'])
        if sr_rir != self.sr:
            rir_audio = resample(rir_audio, orig_freq=sr_rir, new_freq=self.sr)

        rir_audio = rir_audio[0,:]
        v_max = torch.max(torch.abs(rir_audio))
        n0 = torch.argmax(torch.abs(rir_audio))
        rir_audio = rir_audio[n0:] # remove the first n0 samples
        rir_audio = rir_audio / rir_audio[0] # normalize to the first sample
        rir_audio = rir_audio.unsqueeze(0)
        if v_max < 0.05:
            print(f"Index {idx} v_max is too small: {v_max}")
        
        
        # Pad or truncate to target length (16000 samples)
        target_length = self.duration * self.sr
        
        # For rs_audio (reverberant speech)
        if rs_audio.shape[1] > target_length:
            rs_audio = rs_audio[:, :target_length]
        elif rs_audio.shape[1] < target_length:
            rs_audio = torch.nn.functional.pad(rs_audio, (0, target_length - rs_audio.shape[1]))
        
        # For speech_audio (clean speech)
        if speech_audio.shape[1] > target_length:
            speech_audio = speech_audio[:, :target_length]
        elif speech_audio.shape[1] < target_length:
            speech_audio = torch.nn.functional.pad(speech_audio, (0, target_length - speech_audio.shape[1]))
        
        # For rir_audio (RIR)
        if rir_audio.shape[1] > target_length:
            rir_audio = rir_audio[:, :target_length]
        elif rir_audio.shape[1] < target_length:
            rir_audio = torch.nn.functional.pad(rir_audio, (0, target_length - rir_audio.shape[1]))
        
        return rs_audio, speech_audio, rir_audio

def create_dataloader(train_dataset, test_dataset, batch_size=16, shuffle=True, num_workers=18):
    
    # Create separate dataloaders for train and test
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,
        persistent_workers=True
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for test set
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False  # Keep all test samples
    )
    
    return train_dataloader, test_dataloader

def create_dataloader_ddp(train_dataset, test_dataset, batch_size=16, shuffle=True, num_workers=18):
    tr_sampler = DistributedSampler(train_dataset, shuffle=True)
    va_sampler = DistributedSampler(test_dataset, shuffle=False)
    
    # Create separate dataloaders for train and test
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=tr_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,
        persistent_workers=True
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=va_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False  # Keep all test samples
    )
    
    return train_dataloader, test_dataloader

if __name__ == "__main__":
    train_entries = unpickle_data("data/synthetic/dev-real-rir_real_5.pkl")
    test_entries = unpickle_data("data/synthetic/test-real-rir_real_1.pkl")

    train_set = RIR_Dataset(train_entries)
    test_set = RIR_Dataset(test_entries)

    
    train_dataloader, test_dataloader = create_dataloader(train_set, train_set, batch_size=64)
    # print the first batch
    for rs_audio, speech_audio, rir_audio in train_dataloader:
        logger.info(f"Batch shapes: rs_audio={rs_audio.shape}, speech_audio={speech_audio.shape}, rir_audio={rir_audio.shape}")
        break