import os
import torch
import torchaudio
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from train_v2 import SEANetBWEGenerator

# ==========================================
# 1. INITIALIZE DEVICE & MODEL
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading SEANet BWE Generator on {device}...")

model = SEANetBWEGenerator().to(device)

checkpoint_path = "./checkpoints_v2/generator_epoch_10.pth"
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print("Model weights loaded successfully.")
else:
    print(f"WARNING: Checkpoint not found at {checkpoint_path}. Ensure path is correct.")

# ==========================================
# 2. INFERENCE & PLOTTING PIPELINE
# ==========================================
def process_audio(audio_filepath):
    if audio_filepath is None:
        return None, None, None, None
        
    # Load user's audio and ensure mono
    waveform, sr = torchaudio.load(audio_filepath)
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # 1. Create Original Wideband Reference (24kHz)
    if sr != 24000:
        resampler_24k = torchaudio.transforms.Resample(orig_freq=sr, new_freq=24000)
        original_24k = resampler_24k(waveform)
    else:
        original_24k = waveform

    # 2. Create Narrowband Reference (Simulate 8kHz telephonic degradation)
    resampler_8k = torchaudio.transforms.Resample(orig_freq=24000, new_freq=8000)
    narrowband_8k = resampler_8k(original_24k)
    
    # Upsample back to 24kHz geometry for the model input
    resampler_up = torchaudio.transforms.Resample(orig_freq=8000, new_freq=24000)
    input_24k = resampler_up(narrowband_8k)
    
    # 3. Model Inference
    with torch.no_grad():
        reconstructed_24k = model(input_24k.to(device).unsqueeze(0)).squeeze(0).cpu()
        
    # 4. Generate Spectrogram Plot
   # 4. Generate Spectrogram Plot
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000, n_fft=1024, hop_length=256, n_mels=100
    )
    db_transform = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    
    mel_orig = db_transform(mel_transform(original_24k)).squeeze().numpy()
    mel_in = db_transform(mel_transform(input_24k)).squeeze().numpy()
    mel_recon = db_transform(mel_transform(reconstructed_24k)).squeeze().numpy()

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # --- NEW: Precise kHz Mapping for the Mel Axis ---
    # Since n_mels=100 and Nyquist=12000Hz, we map target frequencies to their exact Mel bin index.
    # Formula: bin = (2595 * log10(1 + (Hz / 700))) / max_mel * n_mels
    y_ticks = [0, 47, 66, 87, 99] 
    y_labels = ['0 kHz', '2 kHz', '4 kHz', '8 kHz', '12 kHz']
    
    # Bundle the data and titles for cleaner plotting
    plot_data = [
        (mel_orig, '1. Original Input (Full 24kHz Bandwidth)'),
        (mel_in, '2. Degraded Narrowband (Missing frequencies above 4kHz)'),
        (mel_recon, '3. SEANet Generated Wideband (Reconstructed)')
    ]
    
    for i, (mel_array, title) in enumerate(plot_data):
        axs[i].imshow(mel_array, aspect='auto', origin='lower', cmap='magma')
        axs[i].set_title(title)
        
        # Apply the kHz scale
        axs[i].set_yticks(y_ticks)
        axs[i].set_yticklabels(y_labels)
        axs[i].set_ylabel('Frequency')
        
        # Add a subtle horizontal line at 4kHz to visually highlight the degradation cutoff
        axs[i].axhline(y=66, color='white', linestyle='--', alpha=0.5, linewidth=1)
    
    axs[2].set_xlabel('Time (Frames)')
    
    plt.tight_layout()
    plot_path = "temp_spectrogram.png"
    plt.savefig(plot_path, dpi=150)
    plt.close() # Free memory
    
    # Prepare audio outputs for Gradio (Sample Rate, Numpy Array)
    out_orig = (24000, original_24k.squeeze().numpy())
    out_narrow = (8000, narrowband_8k.squeeze().numpy())
    out_recon = (24000, reconstructed_24k.squeeze().numpy())
    
    return out_orig, out_narrow, out_recon, plot_path

# ==========================================
# 3. GRADIO BLOCKS DASHBOARD
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎙️ Neural Bandwidth Extension (End-to-End 1D GAN)
        Upload a high-quality audio clip or record your voice. The system will internally degrade it to telephonic quality (8kHz), and then use a SEANet-based Generative Adversarial Network to mathematically hallucinate and reconstruct the missing high frequencies.
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_audio = gr.Audio(type="filepath", label="Input Audio")
            process_btn = gr.Button("Generate Wideband Reconstruction", variant="primary")
            
            gr.Markdown(
                """
                ### 📊 Understanding the Spectrogram
                * **Y-Axis (Frequency):** Represents the pitch of the sound (mapped on a human-hearing Mel scale).
                * **X-Axis (Time):** Represents the progression of the audio clip in frames.
                * **Color (Energy):** Bright yellow/orange shows strong frequencies. Dark purple represents silence.
                
                **The Nyquist Limit (Why does it stop at 4kHz?)**
                To digitally record a soundwave, a system needs at least two data points per wave cycle. Therefore, the highest frequency a digital system can capture is always exactly *half* of its sample rate. 
                * **The 4kHz Cutoff:** The degraded telephonic audio (sampled at 8,000 Hz) physically cannot contain sounds above 4kHz. This is why the middle plot goes black above the dashed line.
                * **The 12kHz Reconstruction:** Our SEANet GAN reconstructs the audio at 24,000 Hz, mathematically hallucinating the missing voice frequencies from the 4kHz cutoff all the way up to the new physical limit of 12kHz.
                """
            )
            
        with gr.Column():
            spec_image = gr.Image(type="filepath", label="Mel-Spectrogram Analysis")
            
    with gr.Row():
        audio_orig = gr.Audio(label="1. Original Audio (Ground Truth)")
        audio_narrow = gr.Audio(label="2. Degraded Reference (8kHz)")
        audio_recon = gr.Audio(label="3. SEANet Reconstructed (24kHz)")

    # Connect the button to the processing function
    process_btn.click(
        fn=process_audio,
        inputs=[input_audio],
        outputs=[audio_orig, audio_narrow, audio_recon, spec_image]
    )

if __name__ == "__main__":
    print("Launching Web Dashboard...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)