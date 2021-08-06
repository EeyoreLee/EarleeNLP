# -*- encoding: utf-8 -*-
'''
@create_time: 2021/08/05 14:16:20
@author: lichunyu
'''

from dataclasses import dataclass, field
import logging
import os
import time
import sys
import copy
import math
import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from transformers import TrainingArguments, HfArgumentParser
from transformers.trainer_utils import is_main_process, get_last_checkpoint
import fitlog

from models.flat_bert import Lattice_Transformer_SeqLabel, load_yangjie_rich_pretrain_word_list, equip_chinese_ner_with_lexicon, \
    norm_static_embedding, BertEmbedding, LossInForward, SpanFPreRecMetric, AccuracyMetric, Trainer, FitlogCallback, LRScheduler, \
        LambdaLR, GradientClipCallback, EarlyStopCallback, Callback, WarmupCallback
from utils.common import print_info
from utils.flat.base import load_ner



@dataclass
class CustomizeArguments:

    author: str = field(default="earlee", metadata={"help": "author"})

    update_every: int = field(default=1)

    status: str = field(default='train', metadata={"help": "choice: train | test"})

    use_bert: int = field(default=1)

    only_bert: int = field(default=0)

    fix_bert_epoch: int = field(default=20)

    after_bert: str = field(default='mlp', metadata={"help": "choice: mlp / lstm"})

    msg: str = field(default='11266')

    train_clip: bool = field(default=False, metadata={"help": "是不是要把train的char长度限制在200以内"})

    # debug: int = field(default=0)

    gpumm: bool = field(default=False, metadata={"help": "查看显存"})

    see_convergence: bool = field(default=False)

    see_param: bool = field(default=False)

    test_batch: int = field(default=-1)

    test_train: bool = field(default=False)

    number_normalized: int = field(default=0, metadata={"help": "choice: 0 | 1 | 2 | 3", "help": "0不norm，1只norm char,2norm char和bigram，3norm char，bigram和lattice"})

    lexicon_name: str = field(default='yj', metadata={"help": "choice: lk | yj"})

    use_pytorch_dropout: int = field(default=0)

    char_min_freq: int = field(default=1)

    bigram_min_freq: int = field(default=1)

    lattice_min_freq: int = field(default=1)

    only_train_min_freq: bool = field(default=True)

    only_lexicon_in_train: bool = field(default=False)

    word_min_freq: int = field(default=1)

    early_stop: int = field(default=25)

    epoch: int = field(default=100)

    batch: int = field(default=10)

    optim: str = field(default='sgd', metadata={"help": "choice: sgd | adam | adamw"})

    lr: float = field(default=6e-4)

    bert_lr_rate: float = field(default=0.05)

    embed_lr_rate: float = field(default=1)

    momentum: float = field(default=0.9)

    init: str = field(default='uniform', metadata={"help": "choice: norm | uniform"})

    self_supervised: bool = field(default=False)

    # weight_decay: float = field(default=0.1)

    norm_embed: bool = field(default=True)

    norm_lattice_embed: bool = field(default=True)

    warmup: float = field(default=0.1)

    model: str = field(default='transformer', metadata={"help": "choice: lstm | transformer"})

    lattice: int = field(default=1)

    use_bigram: int = field(default=1)

    hidden: int = field(default=-1)

    ff: int = field(default=3)

    layer: int = field(default=1)

    head: int = field(default=8)

    head_dim: int = field(default=20)

    scaled: bool = field(default=False)

    ff_activate: str = field(default='relu', metadata={"help": "choice: leaky | relu"})

    k_proj: bool = field(default=False)

    q_proj: bool = field(default=True)

    v_proj: bool = field(default=True)

    r_proj: bool = field(default=True)

    attn_ff: bool = field(default=False)

    use_abs_pos: bool = field(default=False)

    use_rel_pos: bool  = field(default=True)

    rel_pos_shared: bool = field(default=True)

    add_pos: bool = field(default=False)

    learn_pos: bool = field(default=False)

    pos_norm: bool = field(default=False)

    rel_pos_init: int = field(default=1)

    four_pos_shared: bool = field(default=True)

    four_pos_fusion: str = field(default='ff_two', metadata={"help": "choice: ff | attn | gate | ff_two | ff_linear, ff就是输入带非线性隐层的全连接，"
                         "attn就是先计算出对每个位置编码的加权，然后求加权和"
                         "gate和attn类似，只不过就是计算的加权多了一个维度"})

    four_pos_fusion_shared: bool = field(default=True, metadata={"help": "是不是要共享4个位置融合之后形成的pos"})

    pre: str = field(default='')

    post: str = field(default='an')

    embed_dropout_before_pos: bool = field(default=False)

    embed_dropout: float = field(default=0.5)

    gaz_dropout: float = field(default=0.5)

    output_dropout: float = field(default=0.3)

    pre_dropout: float = field(default=0.5)

    post_dropout: float = field(default=0.3)

    ff_dropout: float = field(default=0.15)

    ff_dropout_2: float = field(default=-1, metadata={"help": "FF第二层过完后的dropout，之前没管这个的时候是0"})

    attn_dropout: float = field(default=0)

    embed_dropout_pos: str = field(default='0')

    abs_pos_fusion_func: str = field(default='nonlinear_add', metadata={"help": "choice: 'add' |'concat' | 'nonlinear_concat' | 'nonlinear_add' | 'concat_nonlinear' | 'add_nonlinear'"})

    dataset: str = field(default='weibo')



    model_name_or_path: str = field(default='', metadata={"help": "path of PTM"})

    config_name_or_path: str = field(default='')

    tokenizer_name_or_path: str = field(default='')

    num_labels: int = field(default=-1)

    pickle_data_path: str = field(default='')

    test_size: float = field(default=0.2)

    max_length: int = field(default=510)

    deploy: bool = field(default=True, metadata={"help": "save the state_dict of the model if set the field is false"})

    train_pickle_data_path: str = field(default='')

    eval_pickle_data_path: str = field(default='')


def main(json_path):
    yangjie_rich_pretrain_unigram_path = '/root/pretrain-models/flat/gigaword_chn.all.a2b.uni.ite50.vec'
    yangjie_rich_pretrain_bigram_path = '/root/pretrain-models/flat/gigaword_chn.all.a2b.bi.ite50.vec'
    yangjie_rich_pretrain_word_path = '/root/pretrain-models/flat/ctb.50d.vec'
    yangjie_rich_pretrain_char_and_word_path = '/root/pretrain-models/flat/yangjie_word_char_mix.txt'
    # lk_word_path = '/remote-home/xnli/data/pretrain/chinese/sgns.merge.word'
    lk_word_path_2 = '/root/pretrain-models/flat/sgns.merge.word_2'

    load_dataset_seed = 42

    logger = logging.getLogger(__name__)

    fitlog.set_log_dir('logs')

    parser = HfArgumentParser((CustomizeArguments, TrainingArguments))

    if json_path:
        custom_args, training_args = parser.parse_json_file(json_file=json_path)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        custom_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        custom_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    over_all_dropout = -1

    if custom_args.ff_dropout_2 < 0:
        custom_args.ff_dropout_2 = custom_args.ff_dropout

    if over_all_dropout > 0:
        custom_args.embed_dropout = over_all_dropout
        custom_args.output_dropout = over_all_dropout
        custom_args.pre_dropout = over_all_dropout
        custom_args.post_dropout = over_all_dropout
        custom_args.ff_dropout = over_all_dropout
        custom_args.attn_dropout = over_all_dropout

    if custom_args.lattice and custom_args.use_rel_pos:
        custom_args.train_clip = True

    # now_time = get_peking_time()
    if custom_args.test_batch == -1:
        custom_args.test_batch = custom_args.batch//2

    device = torch.device('cuda')

    refresh_data = False

    datasets, vocabs, embeddings = load_ner(
        '/root/hub/golden-horse/data',
        '/root/pretrain-models/flat/gigaword_chn.all.a2b.uni.ite50.vec',
        '/root/pretrain-models/flat/gigaword_chn.all.a2b.bi.ite50.vec',
        _refresh=False,
        index_token=False,
        train_clip=custom_args.train_clip,
        char_min_freq=custom_args.char_min_freq,
        bigram_min_freq=custom_args.bigram_min_freq,
        only_train_min_freq=custom_args.only_train_min_freq
    )


    if custom_args.gaz_dropout < 0:
        custom_args.gaz_dropout = custom_args.embed_dropout

    custom_args.hidden = custom_args.head_dim * custom_args.head
    custom_args.ff = custom_args.hidden * custom_args.ff


    if custom_args.lexicon_name == 'lk':
        yangjie_rich_pretrain_word_path = lk_word_path_2


    w_list = load_yangjie_rich_pretrain_word_list(yangjie_rich_pretrain_word_path,
                                              _refresh=refresh_data,
                                              _cache_fp='cache/{}'.format(custom_args.lexicon_name))

    cache_name = os.path.join('cache',(custom_args.dataset+'_lattice'+'_only_train:{}'+
                          '_trainClip:{}'+'_norm_num:{}'
                                   +'char_min_freq{}'+'bigram_min_freq{}'+'word_min_freq{}'+'only_train_min_freq{}'
                                   +'number_norm{}'+'lexicon_{}'+'load_dataset_seed_{}')
                          .format(custom_args.only_lexicon_in_train,
                          custom_args.train_clip,custom_args.number_normalized,custom_args.char_min_freq,
                                  custom_args.bigram_min_freq,custom_args.word_min_freq,custom_args.only_train_min_freq,
                                  custom_args.number_normalized,custom_args.lexicon_name,load_dataset_seed))


    datasets, vocabs, embeddings = equip_chinese_ner_with_lexicon(datasets,
                                                                vocabs,
                                                                embeddings,
                                                                w_list,
                                                                yangjie_rich_pretrain_word_path,
                                                                _refresh=refresh_data,
                                                                _cache_fp=cache_name,
                                                                only_lexicon_in_train=custom_args.only_lexicon_in_train,
                                                                word_char_mix_embedding_path=yangjie_rich_pretrain_char_and_word_path,
                                                                number_normalized=custom_args.number_normalized,
                                                                lattice_min_freq=custom_args.lattice_min_freq,
                                                                only_train_min_freq=custom_args.only_train_min_freq)


    avg_seq_len = 0
    avg_lex_num = 0
    avg_seq_lex = 0
    train_seq_lex = []
    dev_seq_lex = []
    test_seq_lex = []
    train_seq = []
    dev_seq = []
    test_seq = []
    for k, v in datasets.items():
        max_seq_len = 0
        max_lex_num = 0
        max_seq_lex = 0
        max_seq_len_i = -1
        for i in range(len(v)):
            if max_seq_len < v[i]['seq_len']:
                max_seq_len = v[i]['seq_len']
                max_seq_len_i = i
            # max_seq_len = max(max_seq_len,v[i]['seq_len'])
            max_lex_num = max(max_lex_num,v[i]['lex_num'])
            max_seq_lex = max(max_seq_lex,v[i]['lex_num']+v[i]['seq_len'])

            avg_seq_len+=v[i]['seq_len']
            avg_lex_num+=v[i]['lex_num']
            avg_seq_lex+=(v[i]['seq_len']+v[i]['lex_num'])
            if k == 'train':
                train_seq_lex.append(v[i]['lex_num']+v[i]['seq_len'])
                train_seq.append(v[i]['seq_len'])
                if v[i]['seq_len'] >200:
                    print('train里这个句子char长度已经超了200了')
                    print(''.join(list(map(lambda x:vocabs['char'].to_word(x),v[i]['chars']))))
                else:
                    if v[i]['seq_len']+v[i]['lex_num']>400:
                        print('train里这个句子char长度没超200，但是总长度超了400')
                        print(''.join(list(map(lambda x: vocabs['char'].to_word(x), v[i]['chars']))))
            if k == 'dev':
                dev_seq_lex.append(v[i]['lex_num']+v[i]['seq_len'])
                dev_seq.append(v[i]['seq_len'])
            if k == 'test':
                test_seq_lex.append(v[i]['lex_num']+v[i]['seq_len'])
                test_seq.append(v[i]['seq_len'])


        print('{} 最长的句子是:{}'.format(k,list(map(lambda x:vocabs['char'].to_word(x),v[max_seq_len_i]['chars']))))
        print('{} max_seq_len:{}'.format(k,max_seq_len))
        print('{} max_lex_num:{}'.format(k, max_lex_num))
        print('{} max_seq_lex:{}'.format(k, max_seq_lex))

    max_seq_len = max(* map(lambda x:max(x['seq_len']),datasets.values()))

    show_index = 4
    print('raw_chars:{}'.format(list(datasets['train'][show_index]['raw_chars'])))
    print('lexicons:{}'.format(list(datasets['train'][show_index]['lexicons'])))
    print('lattice:{}'.format(list(datasets['train'][show_index]['lattice'])))
    print('raw_lattice:{}'.format(list(map(lambda x:vocabs['lattice'].to_word(x),
                                    list(datasets['train'][show_index]['lattice'])))))
    print('lex_s:{}'.format(list(datasets['train'][show_index]['lex_s'])))
    print('lex_e:{}'.format(list(datasets['train'][show_index]['lex_e'])))
    print('pos_s:{}'.format(list(datasets['train'][show_index]['pos_s'])))
    print('pos_e:{}'.format(list(datasets['train'][show_index]['pos_e'])))


    for k, v in datasets.items():
        if custom_args.lattice:
            v.set_input('lattice','bigrams','seq_len','target')
            v.set_input('lex_num','pos_s','pos_e')
            v.set_target('target','seq_len')
            v.set_pad_val('lattice',vocabs['lattice'].padding_idx)
        else:
            v.set_input('chars','bigrams','seq_len','target')
            v.set_target('target', 'seq_len')


    if custom_args.norm_embed > 0:
        print('embedding:{}'.format(embeddings['char'].embedding.weight.size()))
        print('norm embedding')
        for k,v in embeddings.items():
            norm_static_embedding(v, custom_args.norm_embed)

    if custom_args.norm_lattice_embed > 0:
        print('embedding:{}'.format(embeddings['lattice'].embedding.weight.size()))
        print('norm lattice embedding')
        for k,v in embeddings.items():
            norm_static_embedding(v,custom_args.norm_embed)


    mode = {}
    mode['debug'] = 0
    mode['gpumm'] = custom_args.gpumm
    # if custom_args.debug or args.gpumm:
    #     fitlog.debug()
    dropout = defaultdict(int)
    dropout['embed'] = custom_args.embed_dropout
    dropout['gaz'] = custom_args.gaz_dropout
    dropout['output'] = custom_args.output_dropout
    dropout['pre'] = custom_args.pre_dropout
    dropout['post'] = custom_args.post_dropout
    dropout['ff'] = custom_args.ff_dropout
    dropout['ff_2'] = custom_args.ff_dropout_2
    dropout['attn'] = custom_args.attn_dropout

    torch.backends.cudnn.benchmark = False
    # fitlog.set_rng_seed(args.seed)
    torch.backends.cudnn.benchmark = False

    bert_embedding = BertEmbedding(vocabs['lattice'],model_dir_or_name='cn-wwm',requires_grad=False,word_dropout=0.01)

    model = Lattice_Transformer_SeqLabel(
        embeddings['lattice'], 
        embeddings['bigram'], 
        custom_args.hidden, 
        len(vocabs['label']),
        custom_args.head, 
        custom_args.layer, 
        custom_args.use_abs_pos,
        custom_args.use_rel_pos,
        custom_args.learn_pos, 
        custom_args.add_pos,
        custom_args.pre, 
        custom_args.post, 
        custom_args.ff, 
        custom_args.scaled,dropout,
        custom_args.use_bigram,
        mode,
        device,vocabs,
        max_seq_len=max_seq_len,
        rel_pos_shared=custom_args.rel_pos_shared,
        k_proj=custom_args.k_proj,
        q_proj=custom_args.q_proj,
        v_proj=custom_args.v_proj,
        r_proj=custom_args.r_proj,
        self_supervised=custom_args.self_supervised,
        attn_ff=custom_args.attn_ff,
        pos_norm=custom_args.pos_norm,
        ff_activate=custom_args.ff_activate,
        abs_pos_fusion_func=custom_args.abs_pos_fusion_func,
        embed_dropout_pos=custom_args.embed_dropout_pos,
        four_pos_shared=custom_args.four_pos_shared,
        four_pos_fusion=custom_args.four_pos_fusion,
        four_pos_fusion_shared=custom_args.four_pos_fusion_shared,
        bert_embedding=bert_embedding
    )



    with torch.no_grad():
        print_info('{}init pram{}'.format('*'*15,'*'*15))
        for n,p in model.named_parameters():
            if 'bert' not in n and 'embedding' not in n and 'pos' not in n and 'pe' not in n \
                    and 'bias' not in n and 'crf' not in n and p.dim()>1:
                try:
                    if custom_args.init == 'uniform':
                        nn.init.xavier_uniform_(p)
                        print_info('xavier uniform init:{}'.format(n))
                    elif custom_args.init == 'norm':
                        print_info('xavier norm init:{}'.format(n))
                        nn.init.xavier_normal_(p)
                except:
                    print_info(n)
                    exit(1208)
        print_info('{}init pram{}'.format('*' * 15, '*' * 15))

    loss = LossInForward()
    encoding_type = 'bmeso'
    if custom_args.dataset == 'weibo':
        encoding_type = 'bio'
    f1_metric = SpanFPreRecMetric(vocabs['label'],pred='pred',target='target',seq_len='seq_len',encoding_type=encoding_type)
    acc_metric = AccuracyMetric(pred='pred',target='target',seq_len='seq_len',)
    acc_metric.set_metric_name('label_acc')
    metrics = [
        f1_metric,
        acc_metric
    ]
    if custom_args.self_supervised:
        chars_acc_metric = AccuracyMetric(pred='chars_pred',target='chars_target',seq_len='seq_len')
        chars_acc_metric.set_metric_name('chars_acc')
        metrics.append(chars_acc_metric)

    if custom_args.see_param:
        for n,p in model.named_parameters():
            print_info('{}:{}'.format(n,p.size()))
        print_info('see_param mode: finish')
        if not custom_args.debug:
            exit(1208)
    datasets['train'].apply
    if custom_args.see_convergence:
        print_info('see_convergence = True')
        print_info('so just test train acc|f1')
        datasets['train'] = datasets['train'][:100]
        if custom_args.optim == 'adam':
            optimizer = optim.AdamW(model.parameters(), lr=custom_args.lr, weight_decay=custom_args.weight_decay)
        elif custom_args.optim == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=custom_args.lr, momentum=custom_args.momentum)
        trainer = Trainer(datasets['train'], model, optimizer, loss, custom_args.batch,
                        n_epochs=custom_args.epoch, dev_data=datasets['train'], metrics=metrics,
                        device=device, dev_batch_size=custom_args.test_batch)

        trainer.train()
        exit(1208)


    bert_embedding_param = list(model.bert_embedding.parameters())
    bert_embedding_param_ids = list(map(id,bert_embedding_param))
    bigram_embedding_param = list(model.bigram_embed.parameters())
    gaz_embedding_param = list(model.lattice_embed.parameters())
    embedding_param = bigram_embedding_param
    if custom_args.lattice:
        gaz_embedding_param = list(model.lattice_embed.parameters())
        embedding_param = embedding_param+gaz_embedding_param
    embedding_param_ids = list(map(id,embedding_param))
    non_embedding_param = list(filter(
        lambda x:id(x) not in embedding_param_ids and id(x) not in bert_embedding_param_ids,
                                        model.parameters()))
    param_ = [{'params': non_embedding_param}, {'params': embedding_param, 'lr': custom_args.lr * custom_args.embed_lr_rate},
                {'params':bert_embedding_param,'lr':custom_args.bert_lr_rate*custom_args.lr}]



    if custom_args.optim == 'adam':
        optimizer = optim.AdamW(param_,lr=custom_args.lr,weight_decay=custom_args.weight_decay)
    elif custom_args.optim == 'sgd':
        # optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,
        #                       weight_decay=args.weight_decay)
        optimizer = optim.SGD(param_,lr=custom_args.lr,momentum=custom_args.momentum,
                            weight_decay=0.)

    if custom_args.dataset == 'msra':
        datasets['dev']  = datasets['test']
    fitlog_evaluate_dataset = {'test':datasets['test']}
    if custom_args.test_train:
        fitlog_evaluate_dataset['train'] = datasets['train']
    evaluate_callback = FitlogCallback(fitlog_evaluate_dataset,verbose=1)
    lrschedule_callback = LRScheduler(lr_scheduler=LambdaLR(optimizer, lambda ep: 1 / (1 + 0.05*ep) ))
    clip_callback = GradientClipCallback(clip_type='value', clip_value=5)

    class Unfreeze_Callback(Callback):
        def __init__(self,bert_embedding,fix_epoch_num):
            super().__init__()
            self.bert_embedding = bert_embedding
            self.fix_epoch_num = fix_epoch_num
            assert self.bert_embedding.requires_grad == False

        def on_epoch_begin(self):
            if self.epoch == self.fix_epoch_num+1:
                self.bert_embedding.requires_grad = True

    callbacks = [
            evaluate_callback,
            lrschedule_callback,
            clip_callback
        ]
    if custom_args.use_bert:
        if custom_args.fix_bert_epoch != 0:
            callbacks.append(Unfreeze_Callback(bert_embedding,custom_args.fix_bert_epoch))
        else:
            bert_embedding.requires_grad = True
    callbacks.append(EarlyStopCallback(custom_args.early_stop))
    if custom_args.warmup > 0 and custom_args.model == 'transformer':
        callbacks.append(WarmupCallback(warmup=custom_args.warmup))


    class record_best_test_callback(Callback):
        def __init__(self,trainer,result_dict):
            super().__init__()
            self.trainer222 = trainer
            self.result_dict = result_dict

        def on_valid_end(self, eval_result, metric_key, optimizer, better_result):
            print(eval_result['data_test']['SpanFPreRecMetric']['f'])

    print(torch.rand(size=[3,3],device=device))

    if custom_args.status == 'train':
        trainer = Trainer(datasets['train'],model,optimizer,loss,custom_args.batch,
                        n_epochs=custom_args.epoch,
                        dev_data=datasets['dev'],
                        metrics=metrics,
                        device=device,callbacks=callbacks,dev_batch_size=custom_args.test_batch,
                        test_use_tqdm=False,check_code_level=-1,
                        update_every=custom_args.update_every)

        trainer.train()



if __name__ == '__main__':
    main('args/empty.json')