import torch
import torch.nn as nn

class ScalableCNN(nn.Module):
    def __init__(self, num_classes=10, width_factor=1, depth=4, in_channels=3, img_size=32):
        """
        Args:
            num_classes: 10 (CIFAR10) or 100 (CIFAR100)
            width_factor: Multiplier for channel size (e.g., 4, 10, 64)
            depth: Number of Conv layers
            in_channels: Number of input channels (3 for RGB, 1 for Grayscale)
        """
        super(ScalableCNN, self).__init__()
        self.layers = nn.ModuleList()
        
        # Base channels configuration
        # in_channels is now passed as argument
        base_channels = 1 
        
        current_channels = in_channels
        current_spatial_dim = img_size # Start with CIFAR-10 size
        
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
                # Only add MaxPool if spatial dim > 1
                if current_spatial_dim > 1:
                    self.layers.append(nn.MaxPool2d(2, 2))
                    current_spatial_dim //= 2
            
            current_channels = out_channels
            
        # คำนวณขนาดก่อนเข้า Fully Connected (Flatten)
        # หมายเหตุ: สำหรับ CIFAR 32x32 ถ้า MaxPool 2 ครั้ง เหลือ 8x8
        # ควรเขียนฟังก์ชันคำนวณอัตโนมัติหากเปลี่ยน Depth บ่อย
        self.flatten_dim = current_channels * (current_spatial_dim ** 2)
        
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