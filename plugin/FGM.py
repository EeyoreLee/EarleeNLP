# -*- encoding: utf-8 -*-
'''
@create_time: 2021/07/13 15:34:42
@author: lichunyu
'''

import torch


class FGM():
    """对抗训练
    """

    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}



        """Usage:
        # 初始化
        fgm = FGM(model)
        for batch_input, batch_label in data:
            # 正常训练
            loss = model(batch_input, batch_label)
            loss.backward() # 反向传播，得到正常的grad
            # 对抗训练
            fgm.attack() # 在embedding上添加对抗扰动
            loss_adv = model(batch_input, batch_label)
            loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            fgm.restore() # 恢复embedding参数
            # 梯度下降，更新参数
            optimizer.step()
            model.zero_grad()
        """