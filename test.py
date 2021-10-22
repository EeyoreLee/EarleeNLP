from datasets import load_dataset
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import torch.nn as nn

test1 = torch.from_numpy(
    np.array(
        [
            [
                [[[1,2,3],
                [2,3,4],
                [6,1,6]]],
                [[[1,2,6],
                [2,3,3],
                [6,1,1]]],
                [[[1,6,3],
                [2,3,7],
                [6,1,9]]],
                [[[1,23,3],
                [2,2,63],
                [6,125,6]]],
                # [[[1,12,3],
                # [2,3,7],
                # [6,77,21]]],
            ]
        ],dtype=np.float32
    )
).transpose(1,2)
test2 = torch.from_numpy(
    np.array(
        [
            [
                [[[1,2,3],
                [2,3,4],
                [6,1,6]]],
                [[[1,2,6],
                [2,3,3],
                [6,1,1]]],
                [[[1,6,3],
                [2,3,7],
                [6,1,9]]],
                [[[1,23,3],
                [2,2,63],
                [6,125,6]]],
                [[[0,0,0],
                [0,0,0],
                [0,0,0]]],
            ]
        ], dtype=np.float32
    )
).transpose(1,2)

conv3d = nn.Conv3d(1,4,3,1,1)
res1 = conv3d(test1)
res2 = conv3d(test2)


pass