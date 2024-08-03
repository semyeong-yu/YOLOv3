import torch
import torch.nn as nn
import torch.nn.functional as F

class Darknet(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Darknet, self).__init__()

        self.relu = nn.ReLU()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(hidden_dim),
            self.relu, 
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(hidden_dim * 2),
            self.relu
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hidden_dim),
            self.relu, 
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(hidden_dim * 2),
            self.relu
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(hidden_dim * 4),
            self.relu
        )

        for i in range(2):
            setattr(self, f'layer4{i}', nn.Sequential(
                nn.Conv2d(hidden_dim * 4, hidden_dim * 2, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(hidden_dim * 2),
                self.relu, 
                nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(hidden_dim * 4),
                self.relu
            ))

        self.layer5 = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(hidden_dim * 8),
            self.relu
        )

        for i in range(8):
            setattr(self, f'layer6{i}', nn.Sequential(
                nn.Conv2d(hidden_dim * 8, hidden_dim * 4, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(hidden_dim * 4),
                self.relu, 
                nn.Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(hidden_dim * 8),
                self.relu
        ))

        self.layer7 = nn.Sequential(
            nn.Conv2d(hidden_dim * 8, hidden_dim * 16, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(hidden_dim * 16),
            self.relu
        )
        
        for i in range(8):
            setattr(self, f'layer8{i}', nn.Sequential(
            nn.Conv2d(hidden_dim * 16, hidden_dim * 8, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hidden_dim * 8),
            self.relu, 
            nn.Conv2d(hidden_dim * 8, hidden_dim * 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(hidden_dim * 16),
            self.relu
        ))

        self.layer9 = nn.Sequential(
            nn.Conv2d(hidden_dim * 16, hidden_dim * 32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(hidden_dim * 32),
            self.relu
        )

        for i in range(4):
            setattr(self, f'layer10{i}', nn.Sequential(
            nn.Conv2d(hidden_dim * 32, hidden_dim * 16, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hidden_dim * 16),
            self.relu, 
            nn.Conv2d(hidden_dim * 16, hidden_dim * 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(hidden_dim * 32),
            self.relu
        ))
        
        self.layer11 = nn.Sequential(
            nn.Conv2d(hidden_dim * 32, 1000, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1000),
            self.relu
        )
        
        self.linear = nn.Linear(1000, 1000, bias=True)

    def forward(self, img):
        # img : shape (N, C, H, W)
        x1 = self.layer1(img) # Conv-Conv
        
        x2 = self.layer2(x1) # Conv-Conv
        x1 = x1 + x2 # res

        x1 = self.layer3(x1) # Conv

        for i in range(2):
            x2 = getattr(self, f'layer4{i}')(x1) # Conv-Conv
            x1 = x1 + x2 # res

        x1 = self.layer5(x1) # Conv
        
        for i in range(8):
            x2 = getattr(self, f'layer6{i}')(x1) # Conv-Conv
            x1 = x1 + x2 # res
        result1 = x1

        x1 = self.layer7(x1) # Conv

        for i in range(8):
            x2 = getattr(self, f'layer8{i}')(x1) # Conv-Conv
            x1 = x1 + x2 # res
        result2 = x1

        x1 = self.layer9(x1)

        for i in range(4):
            x2 = getattr(self, f'layer10{i}')(x1) # Conv-Conv
            x1 = x1 + x2 # res
        result3 = x1

        x1 = self.layer11(x1)
        
        N, C, H, W = x1.shape
        x1 = x1.reshape(N, C, -1).mean(dim=2, keepdim=False) # GAP(Global Average Pooling) : shape (N, C, H*W) -> (N, C)
        
        # shape (N, 1000), (N, 256, 52, 52), (N, 512, 26, 26), (N, 1024, 13, 13)
        return F.softmax(self.linear(x1), dim=1), result1, result2, result3

class FCN(nn.Module):
    def __init__(self, in_dim):
        super(FCN, self).__init__()

        self.relu = nn.ReLU()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim//2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_dim//2),
            self.relu,
            nn.Conv2d(in_dim//2, in_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(in_dim),
            self.relu,
            nn.Conv2d(in_dim, in_dim//2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_dim//2),
            self.relu,
            nn.Conv2d(in_dim//2, in_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(in_dim),
            self.relu,
            nn.Conv2d(in_dim, in_dim//2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_dim//2),
            self.relu,
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_dim//2, in_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(in_dim),
            self.relu,
        )

    def forward(self, feature):
        x1 = self.layer1(feature)
        x2 = self.layer2(x1)
        return x1, x2

class MyYOLO(nn.Module):
    def __init__(self, hidden_dim, n_class):
        super(MyYOLO, self).__init__()

        self.relu = nn.ReLU()

        self.backbone = Darknet(in_dim=3, hidden_dim=hidden_dim)
        
        self.fcn1 = FCN(in_dim=hidden_dim * 24) # output channel : 384, 768
        self.fcn2 = FCN(in_dim=hidden_dim * 32) # output channel : 512, 1024
        self.fcn3 = FCN(in_dim=hidden_dim * 32)
        
        self.detect1 = nn.Sequential(
            nn.Conv2d(hidden_dim * 24, 3 * (4 + 1 + n_class), kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(3 * (4 + 1 + n_class)),
            self.relu
        )
        self.detect2 = nn.Sequential(
            nn.Conv2d(hidden_dim * 32, 3 * (4 + 1 + n_class), kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(3 * (4 + 1 + n_class)),
            self.relu
        )
        self.detect3 = nn.Sequential(
            nn.Conv2d(hidden_dim * 32, 3 * (4 + 1 + n_class), kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(3 * (4 + 1 + n_class)),
            self.relu
        )

    def forward(self, img):
        # shape (N, 1000), (N, 256, 52, 52), (N, 512, 26, 26), (N, 1024, 13, 13)
        _, feature_1, feature_2, feature_3 = self.backbone(img) # obtain 3 feature maps from backbone DarkNet
        
        x1, x2 = self.fcn3(feature_3) # shape (N, 512, 13, 13), (N, 1024, 13, 13)
        detect_3 = self.detect3(x2) # shape (N, 255, 13, 13) for large-scale detection

        x1, x2 = self.fcn2(torch.cat([feature_2, F.interpolate(x1, scale_factor=2, mode='bilinear')], dim=1)) # shape (N, 512, 26, 26), (N, 1024, 26, 26)
        detect_2 = self.detect2(x2) # shape (N, 255, 26, 26) for medium-scale detection

        x1, x2 = self.fcn1(torch.cat([feature_1, F.interpolate(x1, scale_factor=2, mode='bilinear')], dim=1)) # shape (N, 384, 52, 52), (N, 768, 52, 52)
        detect_1 = self.detect1(x2) # shape (N, 255, 52, 52) for small-scale detection

        return detect_1, detect_2, detect_3


