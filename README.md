# EarleeNLP
这是一个个人使用的主要针对于NLP模型训练的仓库，同时提供给任何人自由的使用。  

该仓库后续的优化、改动以及目前正在集成的模型可参考该仓库的 [project](https://github.com/users/EeyoreLee/projects/1)  

如果该仓库的`./trash`文件夹中有对于你有帮助的代码，请fork并移出该文件夹，该仓库会不定期删除`./trash`文件夹下的文件  

如果有任何疑问或需要帮助的地方，请提在issue，issue没有具体的格式要求，你可以随便使用中文或英文

# 使用方法
`main.py` 为训练的入口，有三种方式可以训练  
* `python main.py --output_dir output_dir --epoch 5`
* 编写一个参数文件（json），示例如下，文件路径作为参数传入`main.py`的`main(json_path)`函数 （主要为了debug）
* 编写一个参数文件（json），示例如下，使用`python main.py {json_path}`运行（推荐）

## 参数文件示例
参数参考`./utils/genric.py`中的`AdvanceArguments`，对应模型的`ModelArguments`以及`transformers.TrainingArguments`，`./args`文件夹中有很多历史使用的存档参数供参考
```
{
    "model": "bert_classification",
    "output_dir": "output_dir",
    "author": "test",
    "data_path": "/disk/223/person/lichunyu/workspace/tmp/weibo_3class_200k.csv",
    "model_init_param": {
        "pretrained_model_name_or_path": "/disk/223/person/lichunyu/pretrain-models/bert-base-chinese",
        "num_labels": 3
    },
    "collection_param": {
        "tokenizer_name_or_path": "/disk/223/person/lichunyu/pretrain-models/bert-base-chinese"
    },
    "cuda_visible_devices": "4,5,6,7",
    "num_train_epochs": 5,
    "do_train": true,
    "do_eval": true,
    "per_device_train_batch_size": 64,
    "per_device_eval_batch_size": 64,
    "evaluation_strategy": "epoch",
    "log_level": "info",
    "logging_strategy": "steps",
    "logging_steps": 5,
    "load_best_model_at_end": false,
    "optim": "adamw_hf",
    "group_by_length": false,
    "length_column_name": "length",
    "save_steps": 500,
    "save_total_limit": 5
}
```

## 增加自己的模型
在models文件夹添加以模型名字的“蛇形命名”的文件夹，如下示例中，模型名字为`BertClassification`，文件夹即为`bert_classification`，该文件下需要五个文件，分别为`modeling_{folder_name}`，`argument_{folder_name}`，`collection_{folder_name}`，`trainer_{folder_name}`，`__init__.py`，示例如下
```
.
├── models
│   ├── _base
│   │   ├── base_collection.py
│   │   ├── base_trainer.py
│   │   └── __init__.py
│   ├── bert_classification
│   │   ├── argument_bert_classification.py
│   │   ├── collection_bert_classification.py
│   │   ├── trainer_bert_classification.py
│   │   ├── __init__.py
│   │   └── modeling_bert_classification.py
```
其中  
* `modeling_{folder_name}`文件内放置模型和一个初始化模型的方法，方法命名为`init_func`，该方法的参数由入口传入，并由`argument_{folder_name}`中的`ModelArguments`解析再传入 
* `argument_{folder_name}`文件内放置一个`ModelArgument`用来解析与模型相关的参数，区别与`AdvanceArguments`和`transformers.TrainingArguments` 
* `collection_{folder_name}`文件内放置`Collection`用来接收原始数据并转换为`train_dataset`和`dev_dataset` 
* `trainer_{folder_name}`文件内放置`ModelTrainer`用来继承`transformers.Trainer`并加入自定义的功能，提供了部分常用的任务类型支持，参考_base文件夹
* `__init__.py`仅帮助该文件夹作为一个`module`