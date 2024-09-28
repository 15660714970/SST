import torch
from .MST_Plus_Plus import MST_Plus_Plus
from .Restormer import Restormer
from .MPRNet import MPRNet
from .HSCNN_Plus import HSCNN_Plus
from .hinet import HINet
from .sst import SST


def model_generator(method, pretrained_model_path=None):
    if method == 'mst_plus_plus':
        model = MST_Plus_Plus().cuda()
    elif method == 'restormer':
        model = Restormer().cuda()
    elif method == 'mprnet':
        model = MPRNet(num_cab=4).cuda()
    elif method == 'hscnn_plus':
        model = HSCNN_Plus().cuda()
    elif method == 'hinet':
        model = HINet(depth=4).cuda()
    elif method == 'sst':
        model = SST().cuda()
    else:
        print(f'Method {method} is not defined !!!!')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},
                              strict=True)
    return model
