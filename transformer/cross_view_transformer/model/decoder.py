import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, skip_dim, residual, factor):
        super().__init__()
        # print(f'        DecoderBlock out_channels: {out_channels}')
        dim = out_channels // factor 

        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_channels, 1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels))

        if residual:
            self.up = nn.Conv2d(skip_dim, out_channels, 1) # skip_dim: ${encoder.dim} = 128 , out_channels =[128, 128, 64]
        else:
            self.up = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        # print(f'    Before block: shape {x.shape}')
        x = self.conv(x)

        if self.up is not None:
            # skip is the same input tensor from encoder!
            up = self.up(skip) 
            # F.interpolate: Upsample up.shape[-2:] to becomes x.shape[-2:]
            # Thus, we can add x and up and pass it through a relu()
            up = F.interpolate(up, x.shape[-2:])

            x = x + up
            # print(f'x.shape {x.shape}') # Last block: (1, 64, 200, 200)

        return self.relu(x)


class Decoder(nn.Module):
    def __init__(self, dim, blocks, residual=True, factor=2):
        super().__init__()

        print(f'Instantiate Decoder: dim {dim}\n blocks {blocks}\n residual {residual}\n factor {factor}')
        '''
        From gkt.yaml:
        dim: ${encoder.dim} = 128
        blocks [128, 128, 64]
        residual True
        factor 2
        => dim = 64, 64, 32
        '''

        layers = list()
        in_channels = dim 

        for out_channels in blocks:
            # out_channels = 128, 128, 64
            layer = DecoderBlock(in_channels, out_channels, dim, residual, factor)
            layers.append(layer)
            # this out_chan is the input channel of next DecoderBlock
            in_channels = out_channels

        self.layers = nn.Sequential(*layers)
        self.out_channels = in_channels

    def forward(self, x):
        y = x

        ys = []
        for layer in self.layers:
            y = layer(y, x)
            ys.append(y)
            '''
            Set channel dimensions to be [256, 128, 64]
            Final bev_h/w (the finest): 256
            Decoder: y.shape torch.Size([4, 256, 64, 64])
            Decoder: y.shape torch.Size([4, 128, 128, 128])
            Decoder: y.shape torch.Size([4, 64, 256, 256])     
            '''

        return ys
