import torch.nn as nn


class CrossViewTransformer(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        dim_last: int = 64,
        outputs: dict = {'bev': [0, 1]}
    ):
        super().__init__()

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

    def forward(self, batch):
        x = self.encoder(batch)
        z = x

        # print(f'cvt.py: y.shape  {y.shape}') # shape: (1, 64, 200, 200) is the bev feature!
        # z = self.to_logits(y)

        # print(f'z.shape: {z.shape}') # [1,64,64,64]
        # print(f'cvt.py: z.shape  {z.shape}') # z.shape: torch.Size([1, 64, 200, 200])
        # k =  'bev', [start, stop] = [0, 1]

        bev_dict = {k: z[:, start:stop] for k, (start, stop) in self.outputs.items()}
        # print(f'bev_dict[bev].shape {bev_dict["bev"].shape}') # [1, 1, 64, 64]
        return bev_dict
        return {k: z[:, start:stop] for k, (start, stop) in self.outputs.items()}
