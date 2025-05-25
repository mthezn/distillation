import torch
import torch.nn as nn
import torch.nn.functional as F

class UnetDecoder(nn.Module):
    def __init__(self, encoder_channels=256, out_channels=1):
        super(UnetDecoder, self).__init__()

        self.up1 = self._block(encoder_channels, 128)  # 64 -> 128
        self.up2 = self._block(128, 64)  # 128 -> 256
        self.up3 = self._block(64, 32)  # 256 -> 512
        self.up4 = self._block(32, 16)  # 512 -> 1024

        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up1(x)  # 64x64 -> 128x128
        x = self.up2(x)  # 128x128 -> 256x256
        x = self.up3(x)  # 256x256 -> 512x512
        x = self.up4(x)  # 512x512 -> 1024x1024
        return self.final_conv(x)