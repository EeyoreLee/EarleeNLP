# -*- encoding: utf-8 -*-
'''
@create_time: 2021/08/09 16:54:16
@author: lichunyu
'''
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from utils.flat.base import load_ner
from models.flat_bert import load_yangjie_rich_pretrain_word_list, equip_chinese_ner_with_lexicon
from metircs.functional.f1_score import ner_extract


yangjie_rich_pretrain_word_path = '/root/pretrain-models/flat/ctb.50d.vec'
yangjie_rich_pretrain_char_and_word_path = '/root/pretrain-models/flat/yangjie_word_char_mix.txt'


IDX2LABEL = {
    0: '<pad>', 
    1: '<unk>', 
    2: 'O', 
    3: 'I-Video-Play-name', 
    4: 'I-Radio-Listen-channel', 
    5: 'I-Calendar-Query-datetime_date', 
    6: 'I-Alarm-Update-notes', 
    7: 'I-FilmTele-Play-name', 
    8: 'I-Alarm-Update-datetime_time', 
    9: 'I-Radio-Listen-name', 
    10: 'I-Alarm-Update-datetime_date', 
    11: 'I-HomeAppliance-Control-appliance', 
    12: 'I-Travel-Query-destination', 
    13: 'I-HomeAppliance-Control-details', 
    14: 'I-Radio-Listen-frequency', 
    15: 'I-Music-Play-song', 
    16: 'I-Weather-Query-datetime_date', 
    17: 'B-Calendar-Query-datetime_date', 
    18: 'I-Weather-Query-city', 
    19: 'B-HomeAppliance-Control-appliance', 
    20: 'B-Travel-Query-destination', 
    21: 'I-Video-Play-datetime_date', 
    22: 'I-Music-Play-artist', 
    23: 'B-Alarm-Update-datetime_date', 
    24: 'B-Weather-Query-city', 
    25: 'B-Video-Play-name', 
    26: 'B-Weather-Query-datetime_date', 
    27: 'B-Alarm-Update-datetime_time', 
    28: 'I-FilmTele-Play-artist', 
    29: 'B-Alarm-Update-notes', 
    30: 'B-HomeAppliance-Control-details', 
    31: 'B-FilmTele-Play-name', 
    32: 'B-Radio-Listen-channel', 
    33: 'I-Music-Play-age', 
    34: 'I-FilmTele-Play-age', 
    35: 'B-Radio-Listen-name', 
    36: 'I-FilmTele-Play-tag', 
    37: 'I-Music-Play-album', 
    38: 'B-Music-Play-artist', 
    39: 'B-FilmTele-Play-artist', 
    40: 'I-Travel-Query-departure', 
    41: 'B-Music-Play-song', 
    42: 'I-FilmTele-Play-play_setting', 
    43: 'I-Travel-Query-datetime_date', 
    44: 'B-Travel-Query-departure', 
    45: 'I-Radio-Listen-artist', 
    46: 'B-FilmTele-Play-tag', 
    47: 'I-Travel-Query-datetime_time', 
    48: 'B-Radio-Listen-frequency', 
    49: 'B-Radio-Listen-artist', 
    50: 'I-Video-Play-datetime_time', 
    51: 'B-Video-Play-datetime_date', 
    52: 'B-Travel-Query-datetime_date', 
    53: 'I-FilmTele-Play-region', 
    54: 'B-FilmTele-Play-region', 
    55: 'B-FilmTele-Play-play_setting', 
    56: 'I-TVProgram-Play-name', 
    57: 'B-FilmTele-Play-age', 
    58: 'B-Travel-Query-datetime_time', 
    59: 'B-Music-Play-age', 
    60: 'B-Music-Play-album', 
    61: 'I-Video-Play-region', 
    62: 'B-Video-Play-region', 
    63: 'I-Music-Play-instrument', 
    64: 'I-Weather-Query-datetime_time', 
    65: 'I-TVProgram-Play-channel', 
    66: 'B-Music-Play-instrument', 
    67: 'I-Audio-Play-name', 
    68: 'B-Video-Play-datetime_time', 
    69: 'B-Weather-Query-datetime_time', 
    70: 'B-TVProgram-Play-name', 
    71: 'I-TVProgram-Play-datetime_date', 
    72: 'I-Audio-Play-artist', 
    73: 'B-Audio-Play-name', 
    74: 'I-TVProgram-Play-datetime_time', 
    75: 'B-TVProgram-Play-channel', 
    76: 'B-TVProgram-Play-datetime_date', 
    77: 'B-Audio-Play-artist', 
    78: 'B-TVProgram-Play-datetime_time', 
    79: 'I-Audio-Play-play_setting', 
    80: 'B-Audio-Play-play_setting', 
    81: 'B-Audio-Play-tag', 
    82: 'I-Audio-Play-tag'
}


class NERDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        row = self.dataset[index]
        row['chars'] = torch.tensor(row['chars'])
        row['target'] = torch.tensor(row['target'])
        row['bigrams'] = torch.tensor(row['bigrams'])
        row['seq_len'] = torch.tensor(row['seq_len'])
        row['lex_num'] = torch.tensor(row['lex_num'])
        row['lex_s'] = torch.tensor(row['lex_s'])
        row['lex_e'] = torch.tensor(row['lex_e'])
        row['lattice'] = torch.tensor(row['lattice'])
        row['pos_s'] = torch.tensor(row['pos_s'])
        row['pos_e'] = torch.tensor(row['pos_e'])
        return row



datasets, vocabs, embeddings = load_ner(
    '/root/hub/golden-horse/data',
    '/root/pretrain-models/flat/gigaword_chn.all.a2b.uni.ite50.vec',
    '/root/pretrain-models/flat/gigaword_chn.all.a2b.bi.ite50.vec',
    _refresh=True,
    index_token=False,
    # train_clip=custom_args.train_clip,
    # char_min_freq=custom_args.char_min_freq,
    # bigram_min_freq=custom_args.bigram_min_freq,
    # only_train_min_freq=custom_args.only_train_min_freq
)

w_list = load_yangjie_rich_pretrain_word_list(yangjie_rich_pretrain_word_path,
                                            _refresh=False,
                                            _cache_fp='cache/{}'.format('yj'))


datasets, vocabs, embeddings = equip_chinese_ner_with_lexicon(datasets,
                                                            vocabs,
                                                            embeddings,
                                                            w_list,
                                                            yangjie_rich_pretrain_word_path,
                                                            _refresh=True,
                                                            # _cache_fp=cache_name,
                                                            only_lexicon_in_train=False,
                                                            word_char_mix_embedding_path=yangjie_rich_pretrain_char_and_word_path,
                                                            number_normalized=0,
                                                            lattice_min_freq=1,
                                                            only_train_min_freq=True
                                                            )

def collate_func(batch_dict):
    batch_len = len(batch_dict)
    max_seq_length = max([dic['seq_len'] for dic in batch_dict])
    chars = pad_sequence([i['chars'] for i in batch_dict], batch_first=True)
    target = pad_sequence([i['target'] for i in batch_dict], batch_first=True)
    bigrams = pad_sequence([i['bigrams'] for i in batch_dict], batch_first=True)
    seq_len = torch.tensor([i['seq_len'] for i in batch_dict])
    lex_num = torch.tensor([i['lex_num'] for i in batch_dict])
    lex_s = pad_sequence([i['lex_s'] for i in batch_dict], batch_first=True)
    lex_e = pad_sequence([i['lex_e'] for i in batch_dict], batch_first=True)
    lattice = pad_sequence([i['lattice'] for i in batch_dict], batch_first=True)
    pos_s = pad_sequence([i['pos_s'] for i in batch_dict], batch_first=True)
    pos_e = pad_sequence([i['pos_e'] for i in batch_dict], batch_first=True)
    raw_chars = [i['raw_chars'] for i in batch_dict]
    return [chars, target, bigrams, seq_len, lex_num, lex_s, lex_e, lattice, pos_s, pos_e, raw_chars]

for k, v in datasets.items():
    v.set_input('lattice','bigrams','target', 'seq_len')
    v.set_input('lex_num','pos_s','pos_e')
    v.set_target('target', 'seq_len')
    v.set_pad_val('lattice',vocabs['lattice'].padding_idx)


dev_ds = NERDataset(datasets['test'])
dev_dataloader = DataLoader(
    dev_ds,
    batch_size=1,
    collate_fn=collate_func
)


model = torch.load('/ai/223/person/lichunyu/models/df/ner/flat-2021-08-10-07-50-01-f1_91.pth', map_location=torch.device('cuda'))

model.eval()
for step, batch in enumerate(dev_dataloader):
    #TODO BERT embedding 前20 epoch 冻结
    # chars = batch[0].cuda()
    target = batch[1].cuda()
    bigrams = batch[2].cuda()
    seq_len = batch[3].cuda()
    lex_num = batch[4].cuda()
    # lex_s = batch[5].cuda()
    # lex_e = batch[6].cuda()
    lattice = batch[7].cuda()
    pos_s = batch[8].cuda()
    pos_e = batch[9].cuda()
    raw_chars = batch[10]
    text_list = [''.join(i) for i in raw_chars]

    with torch.no_grad():

        output = model(
            lattice,
            bigrams,
            seq_len,
            lex_num,
            pos_s,
            pos_e,
            target
        )
    pred = output['pred']
    res = ner_extract(pred, seq_len, text_list, idx2label=IDX2LABEL)
    pass

pass