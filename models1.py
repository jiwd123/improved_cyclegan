import torch
import torch.nn as nn
import torch.nn.functional as F
from cbam import CBAMLayer

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.LeakyReLU(0.01,inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(Generator, self).__init__()

        # Initial convolution block       
        self.ICB = nn.Sequential(nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True))

        # Downsampling
        self.down1 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(128),
                        nn.ReLU(inplace=True))
        self.down2 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(256),
                        nn.ReLU(inplace=True))
        self.down3 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(512),
                        nn.ReLU(inplace=True))
        self.down4 = nn.Sequential(nn.Conv2d(512, 1024, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(1024),
                        nn.ReLU(inplace=True))

        self.cbam1 = CBAMLayer(128)
        self.cbam2 = CBAMLayer(256)
        self.cbam3 = CBAMLayer(512)
        self.cbam0 = CBAMLayer(64)

        
        # in_features = 64
        # out_features = in_features*2
        # for _ in range(2):
        #     model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
        #                 nn.InstanceNorm2d(out_features),
        #                 nn.ReLU(inplace=True) ]
        #     in_features = out_features
        #     out_features = in_features*2
            

        # Residual blocks
        self.resblock = nn.Sequential(ResidualBlock(1024))
        # for _ in range(n_residual_blocks):
        #     model += [ResidualBlock(in_features)]

        # Upsampling
        self.PSblock1 = nn.Sequential(nn.PixelShuffle(2),
                        nn.InstanceNorm2d(256),
                        nn.ReLU(inplace=True))
        self.PSblock2 = nn.Sequential(nn.PixelShuffle(2),
                        nn.InstanceNorm2d(128),
                        nn.ReLU(inplace=True))
        self.PSblock3 = nn.Sequential(nn.PixelShuffle(2),
                        nn.InstanceNorm2d(64),
                        nn.ReLU(inplace=True))
        self.PSblock4 = nn.Sequential(nn.PixelShuffle(2),
                        nn.InstanceNorm2d(16),
                        nn.ReLU(inplace=True))

        # out_features = in_features//4
        # for _ in range(2):
        #     model += [  #nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
        #                 #nn.InstanceNorm2d(out_features),
        #                 #nn.LeakyReLU(inplace=True)
        #                 nn.PixelShuffle(2),
        #                 nn.InstanceNorm2d(out_features),
        #                 nn.ReLU(inplace=True)]
        #     in_features = out_features
        #     out_features = in_features//4

        # Output layer
        self.out = nn.Sequential(nn.ReflectionPad2d(3),
                    nn.Conv2d(4, output_nc, 7),
                    nn.Tanh())
        # model += [  nn.ReflectionPad2d(3),
        #             nn.Conv2d(16, output_nc, 7),
        #             nn.Tanh() ]

        #self.model = nn.Sequential(*model)
        self.conv_cat1 = nn.Sequential(
				nn.Conv2d(768, 256, 1, 1, padding=0,bias=True),
				nn.InstanceNorm2d(256),
				nn.ReLU(inplace=True))
        self.conv_cat2 = nn.Sequential(
				nn.Conv2d(320, 64, 1, 1, padding=0,bias=True),
				nn.InstanceNorm2d(64),
				nn.ReLU(inplace=True))
        self.conv_cat3 = nn.Sequential(
				nn.Conv2d(144, 16, 1, 1, padding=0,bias=True),
				nn.InstanceNorm2d(16),
				nn.ReLU(inplace=True))
        self.conv_cat4 = nn.Sequential(
				nn.Conv2d(68, 4, 1, 1, padding=0,bias=True),
				nn.InstanceNorm2d(4),
				nn.ReLU(inplace=True))



    def forward(self, x):
        x0 = self.ICB(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x = self.resblock(x4)
        x = self.resblock(x)
        x = self.resblock(x)
        x = self.resblock(x)
        x = self.resblock(x)
        x = self.resblock(x)
        x = self.resblock(x)
        x = self.resblock(x)
        x = self.resblock(x)
        x = self.PSblock1(x)
        x3 = self.cbam3(x3)
        cat1 = torch.cat([x,x3],dim=1)
        x = self.conv_cat1(cat1)
        x = self.PSblock2(x)
        x2 = self.cbam2(x2)
        cat2 = torch.cat([x,x2],dim=1)
        x = self.conv_cat2(cat2)
        x = self.PSblock3(x)
        x1 = self.cbam1(x1)
        cat3 = torch.cat([x,x1],dim=1)
        x = self.conv_cat3(cat3)
        x = self.PSblock4(x)
        x0 = self.cbam0(x0)
        cat4 = torch.cat([x,x0],dim=1)
        x = self.conv_cat4(cat4)
        x = self.out(x)
        
            
        return x

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)