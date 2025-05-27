import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):  # Reduced from 16 to 8 for stronger attention
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        
        # Initialize weights for better gradient flow
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return torch.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):  # Changed from 7 to 3 for more precise spatial attention
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        # Initialize weights
        nn.init.xavier_uniform_(self.conv.weight)
        
    def forward(self, x):
        # Average pooling along channel axis
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # Max pooling along channel axis
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate along channel axis
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return torch.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8, kernel_size=3):
        super(CBAM, self).__init__()
        self.in_channels = in_channels
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size)
        
    def forward(self, x):
        # Store original input for residual connection
        identity = x
        
        # Apply channel attention
        x = x * self.channel_att(x)
        
        # Apply spatial attention
        x = x * self.spatial_att(x)
        
        # Add residual connection for better gradient flow
        x = x + identity
        
        return x
        
    def __repr__(self):
        return f"CBAM(in_channels={self.in_channels}, reduction_ratio=8, kernel_size=3)"
