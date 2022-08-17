from collections import OrderedDict
import torch
import pickle

from transformers import (
    BertForSequenceClassification
)

# model = torch.load("output_dir/cl_test/checkpoint-200/pytorch_model.bin")
# bert_only = OrderedDict()
# for k, v in model.items():
#     if "bert" in k:
#         bert_only[k] = v
#     elif k == "mlp.dense.weight":
#         bert_only["bert.pooler.dense.weight"] = v
#     elif k == "mlp.dense.bias":
#         bert_only["bert.pooler.dense.bias"] = v
#     else:
#         print(k)

# torch.save(bert_only, "cl_tnews_bert.bin")


bertcls0 = BertForSequenceClassification.from_pretrained("/disk/223/person/lichunyu/pretrain-models/cl-tnews-bert")
# bertcls = BertForSequenceClassification.from_pretrained("/disk/223/person/lichunyu/pretrain-models/bert-base-chinese")
bertcls1 = BertForSequenceClassification.from_pretrained("/disk/223/person/lichunyu/pretrain-models/unsup-simcse-bert-base-uncased")


...