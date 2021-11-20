
from transformers import BertForTokenClassification, BertForMaskedLM

old_func = BertForTokenClassification.__init__

from copy import deepcopy

from torch.nn import *

def __init__(self, *args, **kwargs):
    old_func(self, *args, **kwargs)
    model2 = BertForMaskedLM(*args, **kwargs)
    self.cls = deepcopy(model2.cls)
    del model2

BertForTokenClassification.__init__ = __init__

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
token_prob_of_entity = np.load('token_prob_of_entity.npy').astype(np.float32)
trade_off = 0.0


def forward(
    self,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
):
    outputs = self.bert(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
    )

    sequence_output = outputs[0]

    sequence_output = self.dropout(sequence_output)
    logits = self.classifier(sequence_output)

    outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
    if labels is not None:
        prob_vocal = nn.Softmax(dim=-1)(self.cls(sequence_output)) # new
        token_prob_of_entity_used = torch.from_numpy(token_prob_of_entity).to(sequence_output.device) # new
        loss_fct = CrossEntropyLoss()
        # Only keep active parts of the loss
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
            
            concept_mask = (attention_mask == 1) & (labels > 0) # new
            has_concept = torch.sum(concept_mask.to(torch.int32)).item() != 0 #==== caution! =======
            if has_concept:
                concept_index = ((labels[concept_mask] - 1) / 2.0).to(labels.dtype) # new
                target = token_prob_of_entity_used[concept_index] # new
                output = prob_vocal[concept_mask] # new
                loss_aug = torch.sum(- target * torch.log(output + 1e-10), dim=-1).mean()
                # tqdm.write(str(loss_aug.item()) + ' ' + str(loss.item()))
                loss = loss + trade_off * loss_aug
            # else:
                # tqdm.write(str(loss.item()))
        else:
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        outputs = (loss,) + outputs

    return outputs  # (loss), scores, (hidden_states), (attentions)

BertForTokenClassification.forward = forward
