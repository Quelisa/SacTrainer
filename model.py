from torch import nn
from bert import BertPreTrainedModel, BertModel, BertOnlyMLMHead


class SacBert(BertPreTrainedModel):
    def __init__(self, config, num_labels=1, pooler_type='cls'):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config,
                                   self.bert.embeddings.word_embeddings.weight)
        self.num_labels = num_labels
        classifier_dropout = config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.pooler_type = pooler_type
        assert self.pooler_type in [
            "cls", "avg"
        ], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, input_ids, attention_mask, token_type_ids, forward_mode):
        mode = forward_mode
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}

        if mode == 'mlm':
            hidden_states, _, _ = self.bert(**inputs)
            prediction_scores = self.cls(hidden_states)
            return prediction_scores
        elif mode == 'classify':
            _, _, pooled_output = self.bert(**inputs)
            logits = self.classifier(self.dropout(pooled_output))
            return logits
        elif mode == 'retrieve':
            _, _, pooled_output = self.bert(**inputs)

            last_hidden = pooled_output[:, 0]
            if self.pooler_type in ['cls']:
                return last_hidden[:, 0]
            elif self.pooler_type == "avg":
                return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) /
                        attention_mask.sum(-1).unsqueeze(-1))
            else:
                raise NotImplementedError
