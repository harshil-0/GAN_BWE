import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Import our custom modules
from dataset import LJSpeechBWEDataset
from model import SEANetBWEGenerator, MultiPeriodDiscriminator, MultiResolutionDiscriminator

# =====================================================================
# 4. TRAINING LOOP EXECUTION
# =====================================================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initialized training pipeline on device: {device}")
    
    os.makedirs("./checkpoints", exist_ok=True)
    
    # Hyperparameters from v3
    BATCH_SIZE = 8
    EPOCHS = 10
    LR_GEN = 2e-4
    LR_DISC = 2e-4
    LAMBDA_WAVE = 45.0  
    
    # Dataset Preparation (using the 2.0 override from v3 logic)
    dataset = LJSpeechBWEDataset(data_dir="./data/LJSpeech-1.1", segment_length_sec=2.0)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, _ = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    # Models & Optimizers
    generator = SEANetBWEGenerator().to(device)
    discriminator = MultiPeriodDiscriminator().to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    mrd = MultiResolutionDiscriminator().to(device)
    
    opt_g = optim.AdamW(generator.parameters(), lr=LR_GEN, betas=(0.8, 0.99))
    opt_d = optim.AdamW(list(mpd.parameters()) + list(mrd.parameters()), lr=LR_DISC, betas=(0.8, 0.99))
    
    criterion_gan = nn.MSELoss()
    criterion_wave = nn.L1Loss()
    
    for epoch in range(1, EPOCHS + 1):
        generator.train()
        discriminator.train()
        
        running_g_loss = 0.0
        running_d_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for x_src, y_tgt in progress_bar:
            x_src, y_tgt = x_src.to(device), y_tgt.to(device)
            
            # ----------------------------------------
            # TRAIN DISCRIMINATOR
            # ----------------------------------------
            opt_d.zero_grad()
            
            y_fake = generator(x_src).detach()
            
            res_real_mpd = mpd(y_tgt)
            res_fake_mpd = mpd(y_fake)
            res_real_mrd, _ = mrd(y_tgt)
            res_fake_mrd, _ = mrd(y_fake)
            
            loss_d = 0.0
            for r_real, r_fake in zip(res_real_mpd + res_real_mrd, res_fake_mpd + res_fake_mrd):
                loss_d += criterion_gan(r_real, torch.ones_like(r_real)) + criterion_gan(r_fake, torch.zeros_like(r_fake))
                
            loss_d.backward()
            opt_d.step()
            running_d_loss += loss_d.item()
            
            # ----------------------------------------
            # TRAIN GENERATOR
            # ----------------------------------------
            opt_g.zero_grad()
            
            y_fake = generator(x_src)
            
            res_fake_mpd = mpd(y_fake)
            res_fake_mrd, fmaps_fake_mrd = mrd(y_fake)
            _, fmaps_real_mrd = mrd(y_tgt)
            
            loss_g_adv = 0.0
            for r_fake in res_fake_mpd + res_fake_mrd:
                loss_g_adv += criterion_gan(r_fake, torch.ones_like(r_fake))
                
            loss_fm = 0.0
            for f_real, f_fake in zip(fmaps_real_mrd, fmaps_fake_mrd):
                loss_fm += torch.mean(torch.abs(f_real.detach() - f_fake))
                
            loss_g_wave = criterion_wave(y_fake, y_tgt)
            loss_g_total = loss_g_adv + (LAMBDA_WAVE * loss_g_wave) + (2.0 * loss_fm)
            
            loss_g_total.backward()
            opt_g.step()
            running_g_loss += loss_g_total.item()
            
            progress_bar.set_postfix({
                "G_Loss": f"{loss_g_total.item():.4f}",
                "D_Loss": f"{loss_d.item():.4f}"
            })
            
        epoch_g = running_g_loss / len(train_loader)
        epoch_d = running_d_loss / len(train_loader)
        print(f"--- Epoch {epoch} Summary | Avg G-Loss: {epoch_g:.4f} | Avg D-Loss: {epoch_d:.4f} ---")
        
        torch.save(generator.state_dict(), f"./checkpoints/generator_epoch_{epoch}.pth")

if __name__ == "__main__":
    train()