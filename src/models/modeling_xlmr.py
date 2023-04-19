import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from typing import List, Optional

from transformers import (
    XLMRobertaForSequenceClassification,
    XLMRobertaPreTrainedModel,
    XLMRobertaModel,
    XLMRobertaCongig
)

from .modeling_outputs import QuestionAnsweringNaModelOutput

class XLMRobertaQAAVPool(XLMRobertaPreTrainedModel):

    def __init__(self, config):
        super(XLMRobertaQAAVPool, self).__init__(config)
        self.num_labels = config.num_labels

        self.xlmr = XLMRobertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.has_ans = nn.Sequential(
            nn.Dropout(p=config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 2)
        )

        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        is_impossibles: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.xlmr(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        first_word = sequence_output[:, 0, :]
        has_logits = self.has_ans(first_word)

        total_loss = None
        if (
            start_positions is not None and 
            end_positions is not None and
            is_impossibles is not None
        ):
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            if len(is_impossibles.size()) > 1:
                is_impossibles = is_impossibles.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            is_impossibles = is_impossibles.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)

            cls_loss = loss_fct(has_logits, is_impossibles.long())
            total_loss = (start_loss + end_loss) * 1.0 + cls_loss / 2

        if not return_dict:
            output = (start_logits, end_logits, has_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringNaModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            has_logits=has_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

