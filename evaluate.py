import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi

# Import the new architecture directly from your v2 training script
from train import SEANetBWEGenerator, LJSpeechBWEDataset

# ==========================================
# METRIC CALCULATION FUNCTIONS
# ==========================================
def calculate_lsd(target_audio, generated_audio, n_fft=2048, hop_length=512):
    """Calculates Log-Spectral Distance (LSD) focusing on the frequency domain."""
    window = torch.hann_window(n_fft).to(target_audio.device)
    
    stft_target = torch.stft(target_audio, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    stft_gen = torch.stft(generated_audio, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    
    mag_target = torch.abs(stft_target) + 1e-10
    mag_gen = torch.abs(stft_gen) + 1e-10
    
    log_diff = 20 * torch.log10(mag_target / mag_gen)
    lsd_per_frame = torch.sqrt(torch.mean(log_diff ** 2, dim=0))
    
    return torch.mean(lsd_per_frame).item()

def calculate_pesq_stoi(target_audio, generated_audio, original_sr=24000):
    """Calculates PESQ and STOI. (Both require 16kHz audio for wideband evaluation)."""
    # Detach and move to CPU numpy arrays
    target_np = target_audio.squeeze().cpu().numpy()
    gen_np = generated_audio.squeeze().cpu().numpy()
    
    # Resample to 16000Hz using PyTorch before passing to CPU libraries
    resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=16000).to(target_audio.device)
    target_16k = resampler(target_audio).squeeze().cpu().numpy()
    gen_16k = resampler(generated_audio).squeeze().cpu().numpy()
    
    try:
        # 'wb' stands for wideband (16kHz)
        pesq_score = pesq(16000, target_16k, gen_16k, 'wb')
    except Exception:
        pesq_score = 1.0 # Fallback for silent/corrupt frames
        
    stoi_score = stoi(target_16k, gen_16k, 16000, extended=False)
    
    return pesq_score, stoi_score

# ==========================================
# VISUALIZATION FUNCTION
# ==========================================
def plot_spectrograms(source, target, generated, filename, sr=24000):
    """Generates a 3-panel Mel-Spectrogram for visual inspection."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=1024, hop_length=256, n_mels=100
    )
    db_transform = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    
    mel_src = db_transform(mel_transform(source.cpu())).squeeze().numpy()
    mel_tgt = db_transform(mel_transform(target.cpu())).squeeze().numpy()
    mel_gen = db_transform(mel_transform(generated.cpu())).squeeze().numpy()

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    axs[0].imshow(mel_tgt, aspect='auto', origin='lower', cmap='magma')
    axs[0].set_title('1. Ground Truth Wideband (24kHz)')
    
    axs[1].imshow(mel_src, aspect='auto', origin='lower', cmap='magma')
    axs[1].set_title('2. Narrowband Input (8kHz Upsampled Base)')
    
    axs[2].imshow(mel_gen, aspect='auto', origin='lower', cmap='magma')
    axs[2].set_title('3. SEANet 1D Generated Output (24kHz)')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# ==========================================
# MAIN EVALUATION PIPELINE
# ==========================================
def evaluate(checkpoint_path, num_samples=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Evaluation on {device}...")
    
    # 1. Load the new End-to-End 1D Generator
    model = SEANetBWEGenerator().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # 2. Setup Validation Dataset (Using the exact same split logic)
    full_dataset = LJSpeechBWEDataset(data_dir="./data/LJSpeech-1.1", segment_length_sec=2.0)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    generator_split = torch.Generator().manual_seed(42)
    _, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator_split)
    
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # 3. Evaluation Loop Variables
    total_lsd, total_pesq, total_stoi = 0.0, 0.0, 0.0
    os.makedirs("./evaluation_results", exist_ok=True)
    
    print(f"Evaluating {num_samples} unseen samples from validation split...")
    
    with torch.no_grad():
        for i, (source_audio, target_audio) in enumerate(tqdm(val_loader, total=num_samples)):
            if i >= num_samples:
                break
                
            source_audio = source_audio.to(device)
            target_audio = target_audio.to(device)
            
            # Direct 1D waveform inference pass
            fake_audio = model(source_audio)
            
            # Calculate Metrics (stripping batch dimension via [0])
            lsd = calculate_lsd(target_audio[0], fake_audio[0])
            pesq_val, stoi_val = calculate_pesq_stoi(target_audio[0], fake_audio[0])
            
            total_lsd += lsd
            total_pesq += pesq_val
            total_stoi += stoi_val
            
            # Save visual spectrograms and audio for the first 3 samples
            if i < 3:
                plot_spectrograms(source_audio[0], target_audio[0], fake_audio[0], 
                                  filename=f"./evaluation_results/spectrogram_sample_{i+1}.png")
                torchaudio.save(f"./evaluation_results/target_sample_{i+1}.wav", target_audio[0].cpu(), 24000)
                torchaudio.save(f"./evaluation_results/generated_sample_{i+1}.wav", fake_audio[0].cpu().unsqueeze(0) if fake_audio[0].dim() == 1 else fake_audio[0].cpu(), 24000)

    # 4. Print Final Averages
    avg_lsd = total_lsd / num_samples
    avg_pesq = total_pesq / num_samples
    avg_stoi = total_stoi / num_samples
    
    print("\n" + "="*45)
    print("      FINAL 1D GAN EVALUATION METRICS")
    print("="*45)
    print(f"Log-Spectral Distance (LSD): {avg_lsd:.4f} dB")
    print(f"PESQ (Wideband):             {avg_pesq:.4f} ")
    print(f"STOI:                        {avg_stoi:.4f} ")
    print("="*45)
    print("Saved plots and audio to './evaluation_results/'")

if __name__ == "__main__":
    # Point this to your best saved SEANet epoch
    CHECKPOINT = "./checkpoints/generator_epoch_10.pth"
    
    if os.path.exists(CHECKPOINT):
        evaluate(CHECKPOINT, num_samples=50)
    else:
        print(f"Error: Checkpoint not found at {CHECKPOINT}")