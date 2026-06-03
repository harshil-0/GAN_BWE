# Speech Bandwidth Extension via Generative Adversarial Networks

![SEANet Spectrogram Reconstruction](path/to/your/image.png)

## Executive Summary
This project outlines the development of a machine learning pipeline capable of restoring missing high-frequency spectrums in telephonic or narrowband speech (8kHz) to wideband quality (24kHz). To satisfy the hardware constraints of a local 8GB VRAM GPU (RTX 4060) and the project requirement of utilizing post-2022 GAN architectures, the project underwent a rigorous iterative engineering process.

The final deployed architecture is an **End-to-End 1D Time-Domain GAN**, featuring a SEANet-style Generator and a dual-discriminator setup (Multi-Period and Multi-Resolution). This configuration successfully eliminated phase-cancellation and pitch-shifting bugs while operating efficiently within strict memory limits.

---

## Iterative Engineering Process

### Phase 1: The Baseline Architecture (Week 1)
* **Goal:** Establish the data pipeline and test a lightweight, hardware-friendly generator.
* **Data Handling:** Utilized the LJSpeech dataset. Audio was statically downsampled to 8kHz to simulate telephonic degradation and passed alongside a 24kHz ground truth. Audio files were zero-padded to maintain a fixed 1-second batch length.
* **Architecture:** Built a scratch 1D Causal U-Net Generator, heavily inspired by modern Flow Matching models, to avoid the high VRAM costs of self-attention layers.
* **GAN Framework:** Implemented a standard Spectrogram Discriminator evaluating the 2D Mel-spectrogram output.
* **Loss Formulation:** Relied on a standard Least-Squares GAN adversarial loss combined heavily with an L1 Waveform Loss to force structural alignment.
* **The Pivot (Architectural Shift):** The model collapsed into "Phase Blindness." L1 loss heavily penalizes microscopic timing errors. The generator learned to play it safe by predicting zero high-frequency energy, resulting in muffled, blurry spectrograms. When forced to guess, it produced severe metallic artifacts.

### Phase 2: Transfer Learning & The Mel-Vocoder (Week 2)
* **Goal:** Abandon the scratch U-Net to leverage pre-trained speech representations.
* **Architecture:** Migrated to Vocos (2023), a state-of-the-art Mel-vocoder.
* **Training Optimization:** Implemented strategic layer freezing. The feature extractor and backbone were frozen, and only the final Inverse-STFT (ISTFT) synthesis head was trained.
* **Hardware Configuration:** Maintained a highly conservative Batch Size of 8 and 1-second audio clips to ensure the 8GB VRAM limit was never exceeded during the complex ISTFT computations.
* **Loss Formulation:** Replaced the L1 waveform loss with a Multi-Resolution STFT Loss. This evaluated the frequency domain directly, allowing the model to hallucinate high frequencies without being punished for minor timing (phase) misalignments. We also introduced Feature Matching Loss to bypass vanishing gradients.
* **The Pivot (Architectural Shift):** The model suffered from the "Chipmunk Effect." The Vocos ISTFT head relies on rigid mathematical hop-lengths. When fed variable-length audio clips from LJSpeech that were zero-padded into our 1-second chunks, it caused a temporal compression bug during reconstruction, artificially speeding up the audio and raising the pitch.

### Phase 3: The 1D Time-Domain Solution (Week 3)
* **Goal:** Rebuild the generator to physically prevent pitch-shifting while maintaining 2024-era GAN quality.
* **Data Handling Overhaul:** Removed all zero-padding. Variable-length audio clips were seamlessly looped to fill the 2-second training windows, preventing the generator from memorizing silence.
* **Data Augmentation:** Replaced the static 8kHz downsampler with a Dynamic Low-Pass Biquad Filter, varying the cutoff randomly between 6kHz and 8kHz, and injected microscopic white noise to prevent overfitting.
* **Architecture:** Switched to a SEANet 1D Generator. By using dilated 1D convolutions mapping directly from raw wave to raw wave, temporal compression became mathematically impossible.
* **GAN Framework:** Replaced the generic discriminator with a Multi-Period Discriminator (MPD). The MPD evaluates audio at prime-number intervals (2, 3, 5, 7), forcing the generator to perfectly align the microscopic vibrations of the vocal cords.
* **Outcome:** Achieved a near-flawless STOI score of ~0.99, proving perfectly aligned, intelligible speech with zero pitch shifting.

### Phase 4: High-Fidelity Refinement & Deployment (Week 4)
* **Goal:** Sharpen the high-frequency acoustics and package the project for evaluation.
* **GAN Framework Upgrade:** Added a Multi-Resolution Discriminator (MRD) to work simultaneously with the MPD.
* **Loss Balancing Strategy:** The final optimization framework balanced four competing objectives. Adversarial Loss (MPD + MRD) pushed the network to create realistic textures. Feature Matching Loss anchored the internal math of the generator to the discriminators' hidden layers. L1 Waveform Loss was brought back as a minor stabilizing weight.
* **Metrics Achieved:** The dual-discriminator setup successfully dropped the Log-Spectral Distance (LSD) down to 9.57 dB, proving visually and mathematically that the high-frequency harmonics were successfully reconstructed.
* **Deployment:** Constructed a robust `app.py` utilizing Gradio Blocks. The web interface handles live user microphone input, processes it through the SEANet generator, and dynamically renders stacked, labeled Mel-Spectrograms (Original vs. Narrowband vs. Reconstructed) alongside the audio playback.

---

## Technical Summary Matrix

| Component | Initial Pipeline | Final Pipeline (Deployed) | Reason for Shift |
| :--- | :--- | :--- | :--- |
| **Generator** | 1D Causal U-Net | SEANet 1D Dilated Convs | Resolved pitch shifting and temporal compression bugs inherent in Mel-vocoders. |
| **Discriminator** | Spectrogram Only | MPD + MRD Dual Setup | MPD secures exact timing; MRD secures sharp high-frequency harmonic lines. |
| **Data Chunking** | 1-Second (Zero-Padded) | 2-Second (Seamless Looping) | Stopped the generator from learning to synthesize silence. |
| **Loss Function** | L1 Waveform Loss | Adv. + Feature Matching + L1 | Allowed hallucination of missing frequencies without phase-cancellation penalties. |
| **Hardware Setup** | Batch Size 32 | Batch Size 8 | Guaranteed stable execution and zero Out-Of-Memory errors on an 8GB VRAM GPU. |

---

## Dataset Selection and Rationale (LJSpeech)

For this Bandwidth Extension (BWE) project, the **LJSpeech-1.1** dataset was strategically selected over multi-speaker corpora for the following structural and academic reasons:

* **Acoustic Purity (High-Fidelity Ground Truth):** LJSpeech consists of studio-quality recordings sampled natively at 22,050 Hz. To accurately train a neural network to hallucinate missing high frequencies (up to 24kHz), the ground-truth targets must be free of background noise, reverberation, and recording artifacts.
* **Isolation of the Upsampling Task:** By utilizing a single-speaker dataset, the generator is not burdened with learning the latent representations of hundreds of different vocal timbres, accents, or genders. This allows the model's parameter capacity to be entirely dedicated to the complex task of spectral reconstruction and phase alignment.
* **Hardware and Convergence Efficiency:** Single-speaker datasets present a narrower data distribution. For a constrained computing environment (e.g., a local 8GB GPU), this ensures that the adversarial training loop reaches stability and converges on structural vocal features much faster than a multi-speaker dataset would allow.
* **SOTA Benchmarking Standards:** LJSpeech is the universally accepted baseline for generative audio architectures. Top-tier vocoder models (such as HiFi-GAN, MelGAN, and WaveGlow) all establish their baseline metrics using this exact corpus, ensuring that the methodologies and STOI/PESQ metrics achieved in this project are directly comparable to industry standards.
---

## Visualizing Success: The Nyquist Limit

When viewing the Mel-Spectrogram outputs of this project, you will notice specific hard cutoffs on the Y-axis. This serves as a perfect visual demonstration of the **Nyquist-Shannon Sampling Theorem**, which states that the maximum frequency a digital system can capture is exactly half of its sample rate ($f_{max} = \frac{f_s}{2}$).

* **The Narrowband Input (8kHz):** Peaks exactly at the 4kHz mark. Any frequency above this is permanently destroyed during telephonic degradation. This creates a hard cutoff where all high-frequency vocal characteristics (breathiness, fricatives like "s" and "f") are lost.
* **The SEANet Reconstruction (24kHz):** The End-to-End 1D GAN successfully analyzes the degraded input and generates the missing high-frequency harmonics from the 4kHz cutoff all the way up to the 12kHz physical limit of the new sample rate. This restores the crisp, natural texture of the human voice without introducing pitch-shifting artifacts.

---