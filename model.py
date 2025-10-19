import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size=3,
        padding="same",
        use_batch_norm=True,
        **kwargs,
    ):
        super().__init__()
        use_bias = not use_batch_norm

        self.operation = nn.Sequential(
            nn.Conv2d(
                in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=use_bias
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, X):
        return self.operation(X)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, scale_factor=2, out_size=None):
        super().__init__()

        if out_size:
            self.uplayer = nn.Upsample(out_size, mode="bilinear", align_corners=True)
        else:
            self.uplayer = nn.Upsample(
                scale_factor=scale_factor, mode="bilinear", align_corners=True
            )

        self.convlayer = nn.Sequential(
            ConvBlock(in_ch, in_ch // 2),
            ConvBlock(in_ch // 2, out_ch),
        )

    def forward(self, X, X1):
        X = self.uplayer(X)
        X = self.concatenate_tensors(X, X1)
        return self.convlayer(X)

    @staticmethod
    def concatenate_tensors(X, X1):
        diffY = X1.size()[2] - X.size()[2]
        diffX = X1.size()[3] - X.size()[3]
        X = F.pad(X, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        X = torch.cat([X, X1], dim=1)
        return X


class UNet(nn.Module):
    def __init__(self, chs: list, n_classes: int = 1, **kwargs):
        super().__init__()

        self.depth = len(chs)
        self.enc = nn.Sequential(
            ConvBlock(3, chs[0]),
            ConvBlock(chs[0], chs[0]),
        )
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.out_conv = nn.Conv2d(chs[0], n_classes, kernel_size=1)

        # Create down_blocks and up_blocks based on chs
        for i in range(self.depth - 1):
            is_last_index = i == self.depth - 2

            out_ch = chs[i + 1] // 2 if is_last_index else chs[i + 1]
            self.down_blocks.append(self._create_down_block(chs[i], out_ch))

            out_ch = (
                chs[self.depth - 2 - i]
                if is_last_index
                else chs[self.depth - 2 - i] // 2
            )
            self.up_blocks.append(UpBlock(chs[self.depth - 1 - i], out_ch))

    def forward(self, X):
        enc_features = []
        out = self.enc(X)
        enc_features.append(out)
        # Down path
        for i, down in enumerate(self.down_blocks):
            out = down(out)
            if i < (len(self.down_blocks)) - 1:
                enc_features.append(out)
        # Up path
        for i, up in enumerate(self.up_blocks):
            out = up(out, enc_features[-(i + 1)])
        logits = self.out_conv(out)
        return logits

    def _create_down_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_ch, out_ch),
            ConvBlock(out_ch, out_ch),
        )


if __name__ == "__main__":
    # X = torch.rand(1, 3, 40, 40)
    # logits = model(X)
    # print(logits.shape)
    chs = [32, 64, 128, 256]
    model = UNet(chs, n_classes=5)
    model.to("cuda")
    summary(model, input_size=(3, 196, 140), batch_size=16)
