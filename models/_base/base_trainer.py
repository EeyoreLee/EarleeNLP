# -*- encoding: utf-8 -*-
'''
@create_time: 2022/07/29 16:17:34
@author: lichunyu
'''
from transformers import Trainer
from sklearn.metrics import f1_score, accuracy_score, classification_report, cohen_kappa_score


class BaseTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ClassificationTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def compute_metrics(predictions):
            all_preds = predictions[0].argmax(axis=-1)
            all_labels = predictions[1]
            f1 = f1_score(all_labels, all_preds, average="micro")
            kappa = cohen_kappa_score(all_labels, all_preds)
            return {"f1": f1, "kappa": kappa}

        self.compute_metrics = compute_metrics