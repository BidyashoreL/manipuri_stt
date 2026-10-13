import os
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset

class ManipuriASRDataset(Dataset):
    def __init__(self, csv_path, tokenizer, target_sample_rate=16000, max_audio_length_sec=15.0):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.target_sample_rate = target_sample_rate
        self.max_length = int(target_sample_rate * max_audio_length_sec)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row['audio_path']
        text = row['unified_text']

        # Load audio waveform
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert stereo to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample to 16kHz if necessary
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)

        waveform = waveform.squeeze(0) # Remove channel dimension -> shape (num_samples,)

        # Truncate or pad if audio is too long
        if waveform.shape[0] > self.max_length:
            waveform = waveform[:self.max_length]

        # Tokenize text using Qwen tokenizer
        text_tokens = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=False, 
            truncation=True, 
            max_length=256
        )

        return {
            "input_values": waveform,
            "labels": text_tokens["input_ids"].squeeze(0)
        }