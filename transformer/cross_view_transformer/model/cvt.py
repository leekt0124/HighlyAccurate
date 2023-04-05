import torch.nn as nn

class CrossViewTransformer(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        level: int=1, # single level by default
        decoder_block_channels: list=[256,128,64],
        dim_last: int = 64,
        bev_h: int=128,
        outputs: dict = {'bev': [0, 1]}
    ):
        super().__init__()

        self.level = level
        self.decoder_block_channels = decoder_block_channels

        dim_total = 0
        dim_max = 0

        for _, (start, stop) in outputs.items():
            # start:0, stop:1
            assert start < stop

            dim_total += stop - start
            dim_max = max(dim_max, stop)
            # dim_max = 1

        assert dim_max == dim_total

        self.encoder = encoder
        self.decoder = decoder
        self.outputs = outputs

        print(f'    dim_last:  {dim_last}') # 64
        print(f'    dim_max:   {dim_max}')  # 1
        self.to_logits = nn.Sequential(
            nn.Conv2d(self.decoder.out_channels, dim_last, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim_last),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_last, dim_max, 1))
        
        # A module list
        self.to_logits_list  = nn.ModuleList([
            nn.Sequential(
            nn.Conv2d(self.decoder_block_channels[0], self.decoder_block_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(self.decoder_block_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.decoder_block_channels[0], self.decoder_block_channels[0], 1)),    

            nn.Sequential(
            nn.Conv2d(self.decoder_block_channels[1], self.decoder_block_channels[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(self.decoder_block_channels[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.decoder_block_channels[1], self.decoder_block_channels[1], 1)),     

            nn.Sequential(
            nn.Conv2d(self.decoder_block_channels[2], self.decoder_block_channels[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(self.decoder_block_channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.decoder_block_channels[2], self.decoder_block_channels[2], 1))
            ])    

    def forward(self, batch):
        x = self.encoder(batch)
        ys = self.decoder(x)
        # z = self.to_logits(y)
        zs = []
        for i in range(len(ys)):
            z = self.to_logits_list[i](ys[i])
            zs.append(z) 


        # print(f'cvt.py: z.shape  {z.shape}') # z.shape: torch.Size([1, 64, 200, 200])

        # return a dict of list
        # {"bev": [z1, z2, z3], "center": ...}
        output_dict = {}
        for k, (start,stop) in self.outputs.items():
            output_dict[k] = zs 
        return output_dict
    
        bev_dict = {k: z[:, start:stop] for k, (start, stop) in self.outputs.items()}
        # print(f'bev_dict[bev].shape {bev_dict["bev"].shape}') # [1, 3, 128,  128]
        return bev_dict
        return {k: z[:, start:stop] for k, (start, stop) in self.outputs.items()}
