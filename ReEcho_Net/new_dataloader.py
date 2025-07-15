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

ROOT_PATH = "/home/exouser/re-rir/ReEcho/data" # root path of the dataset

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
    def __init__(self, root=ROOT_PATH, url="dev-clean", sr=16000, duration=2):
        self.libri_speech = LIBRISPEECH(root=root, url=url)
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
    def __init__(self, root=ROOT_PATH, list_file="rir_list_corrected", sr=16000, duration=2):
        self.rir_list = []
        self.parser = ArgumentParser()
        self.parser.add_argument('--rir-id')
        self.parser.add_argument('--room-id')
        self.parser.add_argument('rir_path')
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

        # get rir path
        rir_path = args.rir_path
        rir_path = os.path.join(self.root, rir_path)
        rir, sr = sf.read(rir_path, dtype="float32")
        assert sr == self.sr, "Sample rate mismatch, expected {} but got {}".format(self.sr, sr)
        rir = torch.from_numpy(rir).float()
        rir = rir[:,0] # mono channel
        n0 = torch.argmax(torch.abs(rir))
        rir_out = rir[n0:] / rir[n0]
        # pad or truncate to duration
        rir_out = pad_truncate_norm(rir_out, self.duration * self.sr)
        rir_out = rir_out.unsqueeze(0)
        return rir_out

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
def get_dataloader(audio_dataset, rir_dataset, batch_size=64, pin_memory=False, persistent_workers=False, num_workers=64):
    audio_dataloader = DataLoader(
        audio_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers, drop_last=True
    )
    rir_sampler = AudioSampler(rir_dataset, batch_size * len(audio_dataloader))
    rir_dataloader = DataLoader(
        rir_dataset, batch_size=batch_size, sampler=rir_sampler, num_workers=num_workers, pin_memory=pin_memory,persistent_workers=persistent_workers
        )
    
    return audio_dataloader, rir_dataloader

def get_dataloader_ddp(audio_dataset, rir_dataset, batch_size=64, pin_memory=False, persistent_workers=False, num_workers=64):
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
    
    return audio_dataloader, rir_dataloader




if __name__ == "__main__":
    libri_speech = MyLibriSpeech(url="dev-clean",sr=16000, duration=2)
    print(len(libri_speech))
    rir_dataset = RIRS_Dataset(sr=16000, duration=2)
    print(len(rir_dataset))
    # get dataloader
    ad, rd = get_dataloader(libri_speech, rir_dataset, batch_size=64, num_workers=64)
    print(len(ad), len(rd))

    # get batch data
    for i, (audio, rir) in enumerate(zip(ad, rd)):
        print("batch {}".format(i), audio.shape, rir.shape)
        break









