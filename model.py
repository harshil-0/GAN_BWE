import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

# =====================================================================
# 2. 1D TIME-DOMAIN GENERATOR ARCHITECTURE
# =====================================================================
class DilatedResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(channels, channels, kernel_size=3, padding=1, dilation=1))
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return x + self.conv2(self.lrelu(self.conv1(x)))

class SEANetBWEGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_conv = weight_norm(nn.Conv1d(1, 32, kernel_size=7, padding=3))
        self.res_blocks = nn.Sequential(
            DilatedResidualBlock(32, dilation=1),
            DilatedResidualBlock(32, dilation=3),
            DilatedResidualBlock(32, dilation=9),
            DilatedResidualBlock(32, dilation=27),
        )
        self.post_conv = weight_norm(nn.Conv1d(32, 16, kernel_size=3, padding=1))
        self.final_conv = weight_norm(nn.Conv1d(16, 1, kernel_size=7, padding=3))
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.lrelu(self.init_conv(x))
        x = self.res_blocks(x)
        x = self.lrelu(self.post_conv(x))
        return torch.tanh(self.final_conv(x))

# =====================================================================
# 3. DISCRIMINATOR ARCHITECTURES (MPD & MRD)
# =====================================================================
class PeriodDiscriminator(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(32, 64, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(64, 128, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(128, 256, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(256, 1, (3, 1), 1, padding=(1, 0)))
        ])
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = torch.nn.functional.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        
        for conv in self.convs:
            x = self.lrelu(conv(x))
        return x

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(2),
            PeriodDiscriminator(3),
            PeriodDiscriminator(5),
            PeriodDiscriminator(7),
        ])

    def forward(self, x):
        return [d(x) for d in self.discriminators]
    
class SpectrogramDiscriminator(nn.Module):
    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        window = torch.hann_window(self.win_length).to(x.device)
        stft = torch.stft(x.squeeze(1), self.n_fft, self.hop_length, self.win_length, window=window, return_complex=True)
        mag = torch.abs(stft).unsqueeze(1)
        
        fmaps = []
        for conv in self.convs:
            mag = self.lrelu(conv(mag))
            fmaps.append(mag)
        out = self.conv_post(mag)
        fmaps.append(out)
        return out, fmaps

class MultiResolutionDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            SpectrogramDiscriminator(1024, 256, 1024),
            SpectrogramDiscriminator(2048, 512, 2048),
            SpectrogramDiscriminator(512, 128, 512)
        ])

    def forward(self, x):
        results = []
        fmaps = []
        for d in self.discriminators:
            res, fmap = d(x)
            results.append(res)
            fmaps.extend(fmap)
        return results, fmaps