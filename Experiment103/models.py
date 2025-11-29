import torch
import torch.nn as nn

class ScalableCNN(nn.Module):
    def __init__(self, num_classes=10, width_factor=1, depth=4):
        """
        Args:
            num_classes: 10 (CIFAR10) or 100 (CIFAR100)
            width_factor: Multiplier for channel size (e.g., 4, 10, 64)
            depth: Number of Conv layers
        """
        super(ScalableCNN, self).__init__()
        self.layers = nn.ModuleList()
        
        # Base channels configuration
        in_channels = 3  # RGB Images
        base_channels = 16 
        
        current_channels = in_channels
        
        for i in range(depth):
            # คำนวณ Output Channels ตาม Width Factor
            # สูตร: base * width * (เพิ่มขึ้นเล็กน้อยตามความลึก)
            # ตัวอย่างนี้ใช้แบบ Flat width เพื่อควบคุมตัวแปรให้ง่ายขึ้น
            out_channels = int(base_channels * width_factor)
            
            self.layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1))
            self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(nn.ReLU())
            
            # MaxPool ทุกๆ 2 ชั้น หรือตามความเหมาะสมของขนาดภาพ
            if i % 2 != 0: 
                self.layers.append(nn.MaxPool2d(2, 2))
            
            current_channels = out_channels
            
        # คำนวณขนาดก่อนเข้า Fully Connected (Flatten)
        # หมายเหตุ: สำหรับ CIFAR 32x32 ถ้า MaxPool 2 ครั้ง เหลือ 8x8
        # ควรเขียนฟังก์ชันคำนวณอัตโนมัติหากเปลี่ยน Depth บ่อย
        final_spatial_dim = 32
        num_pool_layers = (depth + 1) // 2  # How many MaxPool layers
        for _ in range(num_pool_layers):
            final_spatial_dim = final_spatial_dim // 2
        self.flatten_dim = current_channels * (final_spatial_dim ** 2)
        
        self.classifier = nn.Linear(self.flatten_dim, num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.classifier(x)
        return x

    def get_num_parameters(self):
        """คืนค่าจำนวนพารามิเตอร์ทั้งหมดในโมเดล"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)