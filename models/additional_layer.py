import torch
import torch.nn as nn

class AdditionalDetectionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(AdditionalDetectionLayer, self).__init__()
        
        # Özellik haritasını yüksek çözünürlüklü hale getirme
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Yüksek çözünürlüklü özellik haritasını işleme
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.SiLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.SiLU()
        
        # Tespit başlığı (Küçük nesneler için özel)
        self.detection_head = nn.Conv2d(out_channels, 3 * (5 + num_classes), kernel_size=1)
        
    def forward(self, x):
        # Özellik haritasını yükselt
        x = self.upsample(x)
        
        # Özellikleri işle
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        # Tespit başlığından çıkış al
        detection = self.detection_head(x)
        
        return detection