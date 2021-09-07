# -*- encoding: utf-8 -*-
'''
@create_time: 2021/09/06 14:11:11
@author: lichunyu
'''
import torch
import torch.nn.functional as F
import numpy as np


def bert_classification_inference(text, model, tokenizer, max_length=150, return_prod=False, label2idx=None, idx2label=None, device='cuda', return_logits=False, manager_dict=None, manager_key=None, **kwargs):
    input_ids = []
    attention_mask = []
    encoded_dict = tokenizer(
                    text, 
                    add_special_tokens = True,
                    truncation='longest_first',
                    max_length = max_length,
                    padding = 'max_length',
                    return_attention_mask = True,
                    return_tensors = 'pt',
            )

    input_ids.append(encoded_dict['input_ids'])
    attention_mask.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)

    model.eval()
    with torch.no_grad():
        input_ids = input_ids.to(device).to(torch.int64)
        attention_mask = attention_mask.to(device).to(torch.int64)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    if idx2label is None:
        idx2label = {v:k for k, v in enumerate(label2idx)}
    classification = idx2label[torch.argmax(output.logits).detach().cpu().numpy().tolist()]
    if manager_dict is not None:
        manager_dict[manager_key] = classification
        return
    if return_logits is True:
        logits = output.logits
        return classification, logits
    return classification