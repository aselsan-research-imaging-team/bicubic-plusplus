import torch.nn as nn


class Bicubic_plus_plus(nn.Module):
    def __init__(self, sr_rate=3):
        super(Bicubic_plus_plus, self).__init__()
        self.conv0 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.conv_out = nn.Conv2d(32, (2*sr_rate)**2 * 3, kernel_size=3, padding=1, bias=False)
        self.Depth2Space = nn.PixelShuffle(2*sr_rate)
        self.act = nn.LeakyReLU(inplace=True, negative_slope=0.1)

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.act(x0)
        x1 = self.conv1(x0)
        x1 = self.act(x1)
        x2 = self.conv2(x1)
        x2 = self.act(x2) + x0
        y = self.conv_out(x2)
        y = self.Depth2Space(y)
        return y
