import torch
import torchaudio
import soundfile as sf
import shlex
import os
from argparse import ArgumentParser
from torchaudio.datasets import LIBRISPEECH
from torch.utils.data import DataLoader, Dataset, Sampler
import torchaudio.functional as F
from torch.utils.data import DistributedSampler

ROOT_PATH = "/home/exouser/re-rir/ReEcho_v2/data" # root path of the dataset

# utils
def pad_truncate_norm(waveform, target_length):
    length = waveform.shape[-1]
    if length > target_length:
        waveform = waveform[..., :target_length]  # clip
    elif length < target_length:
        pad_size = target_length - length
        waveform = torch.nn.functional.pad(waveform, (0, pad_size))  # padding 0
        waveform = waveform / waveform.abs().max()
    return waveform


# LibriSpeech dataset
class MyLibriSpeech(Dataset):
    def __init__(self, root=ROOT_PATH, url="dev-clean", download=False, sr=16000, duration=2):
        self.libri_speech = LIBRISPEECH(root=root, url=url, download=download)
        self.sr = sr
        self.duration = duration
    def __len__(self):
        return len(self.libri_speech)
    def __getitem__(self, index):
        waveform, sr, _, _, _, _ = self.libri_speech[index]
        assert sr == self.sr, "Sample rate mismatch, expected {} but got {}".format(self.sr, sr)
        waveform = pad_truncate_norm(waveform, self.duration * self.sr)
        return waveform
        
        
# RIRS datset
class RIRS_Dataset(Dataset):
    def __init__(self, root=ROOT_PATH, list_file="rir_list_corrected", sr=16000, duration=2, direct_win_ms=2.0):
        self.rir_list = []
        self.parser = ArgumentParser()
        self.parser.add_argument('--rir-id')
        self.parser.add_argument('--room-id')
        self.parser.add_argument('rir_path')
        self.root = root
        self.sr = sr
        self.duration = duration
        self.direct_win = int(direct_win_ms * sr / 1000.0)
        list_file = os.path.join(root, list_file)
        with open(list_file, "r") as f:
            for line in f:
                self.rir_list.append(line.strip())
        
    def __len__(self):
        return len(self.rir_list)
    
    def __getitem__(self, index):
        line = self.rir_list[index].strip()
        args_list = shlex.split(line)
        args = self.parser.parse_args(args_list)

        # get rir path
        rir_path = args.rir_path
        rir_path = os.path.join(self.root, rir_path)
        rir, sr = sf.read(rir_path, dtype="float32")
        assert sr == self.sr, "Sample rate mismatch, expected {} but got {}".format(self.sr, sr)
        rir = torch.from_numpy(rir).float()
        rir = rir[:,0] # mono channel
        n0 = torch.argmax(torch.abs(rir))
        w0 = max(0, n0 - self.direct_win)
        rir_out = rir[w0:] / rir[n0]
        # pad or truncate to duration
        rir_out = pad_truncate_norm(rir_out, self.duration * self.sr)
        rir_out = rir_out.unsqueeze(0)
        return rir_out

# BUT dataset
class BUT_Dataset(Dataset):
    def __init__(self, root=ROOT_PATH, list_file="BUT_RIR_list.txt", sr=16000, duration=2, direct_win_ms=2.0):
        self.but_list = []
        self.sr = sr
        self.duration = duration
        self.direct_win = int(direct_win_ms * sr / 1000.0)
        self.dataset_dir = os.path.join(root, "BUT_ReverbDB")
        list_file = os.path.join(root, list_file)
        with open(list_file, "r") as f:
            for line in f:
                self.but_list.append(line.strip())
        
    def __len__(self):
        return len(self.but_list)
    
    def __getitem__(self, index):
        line = self.but_list[index].strip()
        rir_path = os.path.join(self.dataset_dir, line)
        rir, sr = sf.read(rir_path, dtype="float32")
        assert sr == self.sr, "Sample rate mismatch, expected {} but got {}".format(self.sr, sr)
        rir = torch.from_numpy(rir).float()
        n0 = torch.argmax(torch.abs(rir))
        w0 = max(0, n0 - self.direct_win)
        rir_out = rir[w0:] / rir[n0]
        # pad or truncate to duration
        rir_out = pad_truncate_norm(rir_out, self.duration * self.sr)
        rir_out = rir_out.unsqueeze(0)
        return rir_out

# noise dataset
# RIRS datset
class Noise_Dataset(Dataset):
    def __init__(self, root=ROOT_PATH, list_file="noise_list", sr=16000, duration=2):
        self.rir_list = []
        self.parser = ArgumentParser()
        self.parser.add_argument('--noise-id')
        self.parser.add_argument('--noise-type')
        self.parser.add_argument('--room-linkage')
        self.parser.add_argument('noise_path')
        self.root = root
        self.sr = sr
        self.duration = duration
        list_file = os.path.join(root, list_file)
        with open(list_file, "r") as f:
            for line in f:
                self.rir_list.append(line.strip())
        
    def __len__(self):
        return len(self.rir_list)
    
    def __getitem__(self, index):
        line = self.rir_list[index].strip()
        args_list = shlex.split(line)
        args = self.parser.parse_args(args_list)

        # get noise path
        noise_path = args.noise_path
        noise_path = os.path.join(self.root, noise_path)
        noise, sr = sf.read(noise_path, dtype="float32")
        assert sr == self.sr, "Sample rate mismatch, expected {} but got {}".format(self.sr, sr)
        noise = torch.from_numpy(noise).float()
        noise = noise[:,0] # mono channel
        if noise.shape[-1] > self.duration * self.sr:
            noise = noise[..., :self.duration * self.sr]
        elif noise.shape[-1] < self.duration * self.sr:
            noise = torch.nn.functional.pad(noise, (0, self.duration * self.sr - noise.shape[-1]), mode="circular")
        noise = noise.unsqueeze(0)

        return noise


# Audio Sampler
class AudioSampler(Sampler):
    def __init__(self, data_source, num_samples, replacement=True):
        self.data_source = data_source
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        else:
            result = []
            while len(result) < self.num_samples:
                perm = torch.randperm(n).tolist()
                result.extend(perm)
            return iter(result[:self.num_samples])

    def __len__(self):
        return self.num_samples

# reverbant generate: dataloader
def get_dataloader(audio_dataset, rir_dataset, noise_dataset=None, batch_size=64, pin_memory=False, persistent_workers=False, num_workers=64):
    audio_dataloader = DataLoader(
        audio_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers, drop_last=True
    )
    rir_sampler = AudioSampler(rir_dataset, batch_size * len(audio_dataloader))
    rir_dataloader = DataLoader(
        rir_dataset, batch_size=batch_size, sampler=rir_sampler, num_workers=num_workers, pin_memory=pin_memory,persistent_workers=persistent_workers
        )
    if noise_dataset is not None:
        noise_sampler = AudioSampler(noise_dataset, batch_size * len(audio_dataloader))
        noise_dataloader = DataLoader(
            noise_dataset, batch_size=batch_size, sampler=noise_sampler, num_workers=num_workers, pin_memory=pin_memory,persistent_workers=persistent_workers
            )
        return audio_dataloader, rir_dataloader, noise_dataloader
    else:
        return audio_dataloader, rir_dataloader

def get_dataloader_ddp(audio_dataset, rir_dataset, noise_dataset=None, batch_size=64, pin_memory=False, persistent_workers=False, num_workers=64):
    audio_ddp_sampler = DistributedSampler(audio_dataset, shuffle=True)
    audio_dataloader = DataLoader(
        audio_dataset, batch_size=batch_size, sampler=audio_ddp_sampler, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers, drop_last=True
    )
    #
    total_samples = batch_size * len(audio_dataloader)
    rir_sampler = AudioSampler(rir_dataset, total_samples)
    rir_dataloader = DataLoader(
        rir_dataset, batch_size=batch_size, sampler=rir_sampler, num_workers=num_workers, pin_memory=pin_memory,persistent_workers=persistent_workers
        )
    if noise_dataset is not None:
        noise_sampler = AudioSampler(noise_dataset, total_samples)
        noise_dataloader = DataLoader(
            noise_dataset, batch_size=batch_size, sampler=noise_sampler, num_workers=num_workers, pin_memory=pin_memory,persistent_workers=persistent_workers
            )
        return audio_dataloader, rir_dataloader, noise_dataloader
    else:
        return audio_dataloader, rir_dataloader

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


if __name__ == "__main__":
    libri_speech = MyLibriSpeech(url="dev-clean",sr=16000, duration=2)
    print(len(libri_speech))
    rir_dataset = RIRS_Dataset(sr=16000, duration=2)
    print(len(rir_dataset))
    but_dataset = BUT_Dataset(sr=16000, duration=2)
    print(len(but_dataset))
    noise_dataset = Noise_Dataset(sr=16000, duration=2)
    print(len(noise_dataset))
    # get dataloader
    ad, rd, nd = get_dataloader(libri_speech, but_dataset, noise_dataset, batch_size=64, num_workers=64)
    print(len(ad), len(rd), len(nd))


    # get batch data
    for i, (audio, rir, noise) in enumerate(zip(ad, rd, nd)):
        print("batch {}".format(i), audio.shape, rir.shape, noise.shape)
        snr_db = torch.randint(0, 20, (audio.shape[0], 1), device=audio.device)
        noisy_audio = add_noise(audio, noise, snr_db)
        print("noisy_audio shape: ", noisy_audio.shape)
        break









