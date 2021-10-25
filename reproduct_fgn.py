# -*- encoding: utf-8 -*-
'''
@create_time: 2021/10/25 17:26:35
@author: lichunyu
'''
import time
import datetime

from datasets import load_dataset
from transformers import BertTokenizer, AdamW
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from seqeval.metrics import accuracy_score, f1_score, classification_report

from models.FGN import FGN, CGS_Tokenzier


def tokenize_batch(dataset, tokenizer, cgs_tokenizer,max_length=200, text_name='tokens', label_name='label', **kwargs):
    input_ids = []
    attention_masks = []
    labels = []
    char_input_ids = []
    for i in dataset:
        tokens = [_[0] for _ in i['tokens']]
        encoded_dict = tokenizer(
                            ' '.join(tokens),
                            add_special_tokens = True,
                            truncation='longest_first',
                            max_length = max_length,
                            padding = 'max_length',
                            return_attention_mask = True,
                            return_tensors = 'pt',
                       )

        input_ids.append(encoded_dict['input_ids'])

        attention_masks.append(encoded_dict['attention_mask'])

        labels.append(torch.Tensor(i['ner_tags']))

        char_input_idx = cgs_tokenizer(tokens, max_length=max_length)

        char_input_ids.append(char_input_idx)

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    char_input_ids = torch.cat(char_input_ids, dim=0)
    labels = pad_sequence(labels, batch_first=True)

    tensor_dataset = TensorDataset(input_ids, char_input_ids, attention_masks, labels)
    return tensor_dataset



def format_time(elapsed):

    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def main():
    dataset = load_dataset('weibo_ner')
    train_dataset = dataset['train']
    dev_dataset = dataset['validation']
    global_max_length = 0
    for i in train_dataset['tokens']:
        if len(i) > global_max_length:
            global_max_length = len(i)
    for i in dev_dataset['tokens']:
        if len(i) > global_max_length:
            global_max_length = len(i)
    print("global_max_length is {}".format(global_max_length))

    all_label = []
    for i in train_dataset['ner_tags']:
        all_label.extend(i)
    label_size = len(list(set(all_label)))
    print("label size is {}".format(label_size))

    max_length = 200
    bert_model_path = '/ai/223/person/lichunyu/pretrain-models/bert-base-chinese'

    tokenizer = BertTokenizer.from_pretrained('/ai/223/person/lichunyu/pretrain-models/bert-base-chinese')

    cgs_tokenizer = CGS_Tokenzier.from_pretained('/root/EarleeNLP')

    train_tensor_dataset = tokenize_batch(train_dataset, tokenizer, cgs_tokenizer)
    dev_tensor_data = tokenize_batch(dev_dataset, tokenizer, cgs_tokenizer)

    train_dataloader = DataLoader(train_tensor_dataset, sampler=RandomSampler(train_tensor_dataset), batch_size=16)
    dev_dataloader= DataLoader(dev_tensor_data, sampler=SequentialSampler(dev_tensor_data), batch_size=16)

    weights = torch.load('fgn_weights_gray_with_pad.pth')
    model = FGN(bert_model_path, weights, label_size=label_size)
    model = model.cuda()
    optimizer = AdamW(model.parameters(),
                  lr = 5e-5,
                  eps = 1e-6
                )
    epoch = 50

    for i in epoch:

        bt = time.time()
        # total_train_loss = 0
        # model.train()

        # for step, batch in enumerate(train_dataloader):
        #     if step % 50 == 0 and not step == 0:
        #         elapsed = format_time(time.time() - bt)
        #         print('  Batch {:>5,}  of  {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        #     model.zero_grad()

        #     input_ids = batch[0].cuda()
        #     char_input_ids = batch[1].cuda()
        #     attention_masks = batch[2].cuda()
        #     label = batch[3].cuda()

        #     output = model(input_ids, char_input_ids, attention_masks, label=label)
        #     loss = output['loss']
        #     total_train_loss += loss.item()
        #     loss.backward()
        #     optimizer.step()
        # print('train loss is {}'.format(total_train_loss))


        model.eval()
        total_dev_loss = 0
        label_pred = []
        label_gt = []
        for step, batch in enumerate(dev_dataloader):

            input_ids = batch[0].cuda()
            char_input_ids = batch[1].cuda()
            attention_masks = batch[2].cuda()
            label = batch[3].cuda()

            with torch.no_grad():
                output = model(input_ids, char_input_ids, attention_masks, label=label)
                loss = output['loss']
                total_dev_loss += loss.item()
                label_pred.extend(output['pred'])
                label_gt.extend(label)

        pass

    pass


if __name__ == '__main__':
    main()
