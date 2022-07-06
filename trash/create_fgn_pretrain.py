import torch
import numpy as np



ckpt = torch.load('fgn_weights_gray.pth')
pad = torch.zeros((1,2500))
pth = torch.cat([ckpt, pad], dim=0)
torch.save(pth, 'fgn_weights_gray_with_pad.pth')
pass