# src/model.py
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(Convolution => BatchNorm => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class GraphCVNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # Encoder (Downsampling path)
        self.inc = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down1 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down2 = DoubleConv(128, 256)
        
        # Bottleneck
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(256, 512)
        
        # Decoder (Upsampling path)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256) # 256 (from up1) + 256 (skip connection) = 512
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128) # 128 + 128 = 256
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)   # 64 + 64 = 128
        
        # Final output layer (spits out raw logit values for edge probability)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)         # Resolution: 256x256, Channels: 64
        x2 = self.down1(self.pool1(x1)) # Resolution: 128x128, Channels: 128
        x3 = self.down2(self.pool2(x2)) # Resolution: 64x64,   Channels: 256
        
        # Bottleneck
        b = self.bottleneck(self.pool3(x3)) # Resolution: 32x32, Channels: 512
        
        # Decoder with Skip Connections
        # We upsample the deeper features, then concatenate them with the high-res 
        # features saved directly from the encoder stage (x1, x2, x3)
        u1 = self.up1(b)
        u1 = torch.cat([u1, x3], dim=1)
        u1 = self.conv_up1(u1)
        
        u2 = self.up2(u1)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.conv_up2(u2)
        
        u3 = self.up3(u2)
        u3 = torch.cat([u3, x1], dim=1)
        u3 = self.conv_up3(u3)
        
        logits = self.outc(u3)
        return logits

if __name__ == "__main__":
    print("GraphCVNet architecture file loaded.")