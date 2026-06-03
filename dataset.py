import os
import torch
import torchaudio
from torch.utils.data import Dataset

# =====================================================================
# 1. BULLETPROOF DATASET LOADER
# =====================================================================
class LJSpeechBWEDataset(Dataset):
    def __init__(self, data_dir, segment_length_sec=2.0, target_sr=24000, native_low_sr=8000):
        self.data_dir = os.path.join(data_dir, "wavs")
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Missing LJSpeech wavs directory at {self.data_dir}")
            
        self.audio_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith(".wav")]
        self.target_sr = target_sr
        self.native_low_sr = native_low_sr
        self.target_length = int(segment_length_sec * target_sr)
        
        # Resampling transforms for creating input pairs
        self.downsample = torchaudio.transforms.Resample(orig_freq=target_sr, new_freq=native_low_sr)
        self.upsample = torchaudio.transforms.Resample(orig_freq=native_low_sr, new_freq=target_sr)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.audio_files[idx])
        
        if sr != self.target_sr:
            resample_fn = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resample_fn(waveform)
            
        waveform = waveform.squeeze(0)
        
        # FIX: Loop the audio if it is shorter than the required segment window
        if waveform.size(0) < self.target_length:
            repeats = (self.target_length // waveform.size(0)) + 2
            waveform = waveform.repeat(repeats)
            
        # Deterministic slice window via random crop
        max_start = waveform.size(0) - self.target_length
        start = torch.randint(0, max_start + 1, (1,)).item()
        target_wave = waveform[start:start + self.target_length]
            
        # Create the degraded narrowband input upsampled back to 24kHz geometry
        source_wave_8k = self.downsample(target_wave.unsqueeze(0))
        source_wave_24k = self.upsample(source_wave_8k).squeeze(0)
            
        return source_wave_24k.unsqueeze(0), target_wave.unsqueeze(0)