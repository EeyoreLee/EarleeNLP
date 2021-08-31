# -*- encoding: utf-8 -*-
'''
@create_time: 2021/08/05 14:28:18
@author: lichunyu
'''

from dataclasses import dataclass, field



@dataclass
class CustomizeArguments:

    author: str = field(default="earlee", metadata={"help": "author"})

    description: str = field(default='')

    log_file_path: str = field(default='logs/log.log')




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

    lr: float = field(default=5e-5)

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

    weight_start: float = field(default=1.)

    weight_end: float = field(default=1.)

    weight_span: float = field(default=0.7)







def print_info(*inp,islog=True,sep=' '):
    from fastNLP import logger
    if islog:
        print(*inp,sep=sep)
    else:
        inp = sep.join(map(str,inp))
        logger.info(inp)


def text_rm_space(text:str):
    offsets, pointer = [], 0
    for idx, char in enumerate(text):
        offsets.append((pointer, idx))
        if char != ' ':
            pointer += 1
    return text.replace(' ', ''), offsets




if __name__ == '__main__':
    print(text_rm_space('as fas ,fsfsf asdf'))