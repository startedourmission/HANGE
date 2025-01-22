import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, 1)
        self.key = nn.Conv2d(in_channels, in_channels//8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        query = self.query(x).view(B, -1, H*W).permute(0, 2, 1)  # B, HW, C'
        key = self.key(x).view(B, -1, H*W)  # B, C', HW
        value = self.value(x).view(B, -1, H*W)  # B, C, HW

        attention = self.softmax(torch.bmm(query, key))  # B, HW, HW
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B, C, HW
        out = out.view(B, C, H, W)
        return self.gamma * out + x

class Generator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, attention_position='middle'):
        super().__init__()
        self.attention_position = attention_position
        
        # Initial conv (3 -> 64)
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 7, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )

        # Downsampling (64 -> 128 -> 256)
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        )

        # Residual blocks (256 -> 256)
        self.res_blocks = nn.ModuleList([
            ResBlock(256) for _ in range(4)
        ])

        # Upsampling (256 -> 128 -> 64)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),e
            nn.ReLU(True)
        )

        # Output conv (64 -> 3)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, output_channels, 7, padding=3, padding_mode='reflect'),
            nn.Tanh()
        )

        # 위치별 어텐션 모듈 생성
        if attention_position == 'early':
            self.attention = SelfAttention(64)
        elif attention_position == 'middle':
            self.attention = SelfAttention(256)
        else:  # late
            self.attention = SelfAttention(64)

    def forward(self, x):
        # Initial convolution (3 -> 64)
        x = self.conv1(x)
        
        # Early attention
        if self.attention_position == 'early':
            x = self.attention(x)  # 64 channels
        
        # Downsampling
        x = self.down1(x)  # 64 -> 128
        x = self.down2(x)  # 128 -> 256
        
        # Middle attention
        if self.attention_position == 'middle':
            x = self.attention(x)  # 256 channels
        
        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)  # 256 channels
            
        # Upsampling
        x = self.up1(x)  # 256 -> 128
        x = self.up2(x)  # 128 -> 64
        
        # Late attention
        if self.attention_position == 'late':
            x = self.attention(x)  # 64 channels
            
        return self.conv2(x)  # 64 -> 3

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        
        # 70x70 PatchGAN
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(256, 512, 4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.net(x)

# Loss functions
class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):
        target_tensor = self.real_label if target_is_real else self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)

# Training loop
def train_step(real_A, real_B, netG, netD, criterionGAN, optimG, optimD):
    # Update D
    optimD.zero_grad()
    fake_B = netG(real_A)
    
    # Real
    pred_real = netD(real_B)
    loss_D_real = criterionGAN(pred_real, True)
    
    # Fake
    pred_fake = netD(fake_B.detach())
    loss_D_fake = criterionGAN(pred_fake, False)
    
    loss_D = (loss_D_real + loss_D_fake) * 0.5
    loss_D.backward()
    optimD.step()

    # Update G
    optimG.zero_grad()
    pred_fake = netD(fake_B)
    loss_G = criterionGAN(pred_fake, True)
    loss_G.backward()
    optimG.step()

    return {'loss_D': loss_D.item(), 'loss_G': loss_G.item()}

# Example usage
def create_model(attention_position='middle'):
    return {
        'G': Generator(attention_position=attention_position),
        'D': Discriminator()
    }