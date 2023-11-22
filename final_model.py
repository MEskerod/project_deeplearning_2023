#TO IMPORT MODEL CLASS USE import RNA_Unet


from torchvision import transforms

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
The layers and functions that are used in the model
"""

class DynamicPadLayer(nn.Module):
  def __init__(self, stride_product):
    super(DynamicPadLayer, self).__init__()
    self.stride_product = stride_product

  def forward(self, x):
    input_size = x.shape[2]
    padding = self.calculate_padding(input_size, self.stride_product)
    return nn.functional.pad(x, padding)

  def calculate_padding(self, input_size, stride_product):
    p = stride_product - input_size % stride_product
    return (0, p, 0, p)

class AveragePooling(nn.Module):
  """
  Layer for average pooling
  """
  def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0):
    super(AveragePooling, self).__init__()
    self.avg_pool = nn.AvgPool2d(kernel_size = kernel_size, stride = stride)

  def forward(self, x):
    return self.avg_pool(x)

class RNA_Unet(nn.Module):
    def __init__(self, channels=64, in_channels=4, output_channels=4, activation = F.leaky_relu, pooling = AveragePooling):
        """
        args:
        channels: number of channels in the first hidden layers of both encoder
        in_channels: number of channels in input images (RGBA)
        output_channels: number of channels in output image
        activation: activation function
        pooling: type of pooling that is used for pooling layers
        """
        super(RNA_Unet, self).__init__()

        self.activation = activation

        self.pad = DynamicPadLayer(2**4)

        # Encoder
        self.bn11 = nn.BatchNorm2d(channels)
        self.e11 = nn.Conv2d(in_channels, channels, kernel_size = 3, padding = 1)
        self.bn12 = nn.BatchNorm2d(channels)
        self.e12 = nn.Conv2d(channels, channels, kernel_size = 3, padding = 1)
        self.bn13 = nn.BatchNorm2d(channels)
        self.pool1 = pooling(channels, channels, kernel_size=2, stride=2)

        self.bn21 = nn.BatchNorm2d(channels * 2)
        self.e21 = nn.Conv2d(channels, channels*2, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(channels * 2)
        self.e22 = nn.Conv2d(channels*2, channels*2, kernel_size=3, padding=1)
        self.bn23 = nn.BatchNorm2d(channels * 2)
        self.pool2 = pooling(channels*2, channels*2, kernel_size=2, stride=2)

        self.bn31 = nn.BatchNorm2d(channels*4)
        self.e31 = nn.Conv2d(channels*2, channels*4, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(channels*4)
        self.e32 = nn.Conv2d(channels*4, channels*4, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(channels*4)
        self.pool3 = pooling(channels*4, channels*4, kernel_size=2, stride=2)

        self.bn41 = nn.BatchNorm2d(channels*8)
        self.e41 = nn.Conv2d(channels*4, channels*8, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(channels*8)
        self.e42 = nn.Conv2d(channels*8, channels*8, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(channels*8)
        self.pool4 = pooling(channels*8, channels*8, kernel_size=2, stride=2)

        self.bn51 = nn.BatchNorm2d(channels*16)
        self.e51 = nn.Conv2d(channels*8, channels*16, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(channels*16)
        self.e52 = nn.Conv2d(channels*16, channels*16, kernel_size=3, padding=1)

        #Decoder
        self.bn61 = nn.BatchNorm2d(channels*8)
        self.upconv1 = nn.ConvTranspose2d(channels*16, channels*8, kernel_size=2, stride=2)
        self.bn62 = nn.BatchNorm2d(channels*8)
        self.d11 = nn.Conv2d(channels*16, channels*8, kernel_size=3, padding=1)
        self.bn63 = nn.BatchNorm2d(channels*8)
        self.d12 = nn.Conv2d(channels*8, channels*8, kernel_size=3, padding=1)

        self.bn71 = nn.BatchNorm2d(channels*4)
        self.upconv2 = nn.ConvTranspose2d(channels*8, channels*4, kernel_size=2, stride=2)
        self.bn72 = nn.BatchNorm2d(channels*4)
        self.d21 = nn.Conv2d(channels*8, channels*4, kernel_size=3, padding=1)
        self.bn73 = nn.BatchNorm2d(channels*4)
        self.d22 = nn.Conv2d(channels*4, channels*4, kernel_size=3, padding=1)

        self.bn81 = nn.BatchNorm2d(channels*2)
        self.upconv3 = nn.ConvTranspose2d(channels*4, channels*2, kernel_size=2, stride=2)
        self.bn82 = nn.BatchNorm2d(channels*2)
        self.d31 = nn.Conv2d(channels*4, channels*2, kernel_size=3, padding=1)
        self.bn83 = nn.BatchNorm2d(channels*2)
        self.d32 = nn.Conv2d(channels*2, channels*2, kernel_size=3, padding=1)

        self.bn91 = nn.BatchNorm2d(channels)
        self.upconv4 = nn.ConvTranspose2d(channels*2, channels, kernel_size=2, stride=2)
        self.bn92 = nn.BatchNorm2d(channels)
        self.d41 = nn.Conv2d(channels*2, channels, kernel_size=3, padding=1)
        self.bn93 = nn.BatchNorm2d(channels)
        self.d42 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.out = nn.Sequential(nn.Conv2d(channels, output_channels, kernel_size=3, padding=1),
                                 nn.Sigmoid())

        # Initialize weights
        self.init_weights()

    def init_weights(self):
      for layer in self.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
          gain = nn.init.calculate_gain("leaky_relu", 0.01)
          nn.init.xavier_uniform_(layer.weight, gain=gain)
          nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.BatchNorm2d):
          nn.init.constant_(layer.weight, 1)
          nn.init.constant_(layer.bias, 0)


    def forward(self, x):
        dim = x.shape[2]
        x = self.pad(x)

        #Encoder
        xe11 = self.bn11(self.activation(self.e11(x)))
        xe12 = self.bn12(self.activation(self.e12(xe11)))
        xp1 = self.bn13(self.pool1(xe12))

        xe21 = self.bn21(self.activation(self.e21(xp1)))
        xe22 = self.bn22(self.activation(self.e22(xe21)))
        xp2 = self.bn23(self.pool2(xe22))

        xe31 = self.bn31(self.activation(self.e31(xp2)))
        xe32 = self.bn32(self.activation(self.e32(xe31)))
        xp3 = self.bn33(self.pool3(xe32))

        xe41 = self.bn41(self.activation(self.e41(xp3)))
        xe42 = self.bn42(self.activation(self.e42(xe41)))
        xp4 = self.bn43(self.pool4(xe42))

        xe51 = self.bn51(self.activation(self.e51(xp4)))
        xe52 = self.bn52(self.activation(self.e52(xe51)))

        #Decoder
        xu1 = self.bn61(self.activation(self.upconv1(xe52)))
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = self.bn62(self.activation(self.d11(xu11)))
        xd12 = self.bn63(self.activation(self.d12(xd11)))

        xu2 = self.bn71(self.activation(self.upconv2(xd12)))
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = self.bn72(self.activation(self.d21(xu22)))
        xd22 = self.bn73(self.activation(self.d22(xd21)))

        xu3 = self.bn81(self.activation(self.upconv3(xd22)))
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = self.bn81(self.activation(self.d31(xu33)))
        xd32 = self.bn83(self.activation(self.d32(xd31)))

        xu4 = self.bn91(self.activation(self.upconv4(xd32)))
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = self.bn92(self.activation(self.d41(xu44)))
        xd42 = self.bn93(self.activation(self.d42(xd41)))

        out = self.out(xd42)

        out = out[:, :, :dim, :dim]

        return out
    

"""
BELOW IS THE DATA TRANSFORMER USED TO PREPARE IMAGES FOR THE MODEL
"""

dataTransformer = transforms.Compose([
    transforms.ToTensor(), #Convert image to tensor
])
