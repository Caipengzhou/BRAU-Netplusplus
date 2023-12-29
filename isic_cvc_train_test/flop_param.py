import platform
import pathlib
plt = platform.system()
if plt != 'Windows':
  pathlib.WindowsPath = pathlib.PosixPath
from networks.bra_unet import BRAUnet
from thop import profile
import torch
net = BRAUnet(img_size=256, in_chans=3, num_classes=1, n_win=8)
net.load_from()
randn_input = torch.randn(1, 3, 256, 256)
flops, params = profile(net, inputs=(randn_input, ))
print('FLOPs = ' + str(flops/1000**3) + 'G')
print('Params = ' + str(params/1000**2) + 'M')
