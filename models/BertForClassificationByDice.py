# -*- encoding: utf-8 -*-
'''
@create_time: 2021/07/12 17:20:46
@author: lichunyu
'''

import torch.nn as nn
from transformers import (
    BertModel,
    BertConfig,
    BertPreTrainedModel,
    BertForSequenceClassification
)
from transformers.modeling_outputs import (
    SequenceClassifierOutput
)

from loss.dice_loss import DiceLoss
from loss.focal_loss import FocalLoss



class BertForClassificationByDice(BertPreTrainedModel):
    """
    Just for single label classification.
    Not support regression and multi label classification.
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                self.config.problem_type = 'single_label_classification'
            elif self.config.problem_type != 'single_label_classification':
                raise NotImplementedError(self.__doc__)

            if self.config.problem_type == 'single_label_classification':
                # loss_fct = DiceLoss()
                loss_fct = FocalLoss(gamma=2, alpha=[1,3,3,3,3,3,3], reduction='sum')
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
