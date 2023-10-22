import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from transformers import RobertaModel


BIAS_TERMS_DICT = {
    'intermediate': 'intermediate.dense.bias',
    'key': 'attention.self.key.bias',
    'query': 'attention.self.query.bias',
    'value': 'attention.self.value.bias',
    'output': 'output.dense.bias',
    'output_layernorm': 'output.LayerNorm.bias',
    'attention_layernorm': 'attention.output.LayerNorm.bias',
    'all': 'bias',
}


def convert_to_actual_components(components):
    return [BIAS_TERMS_DICT[component] for component in components]

def perform_bitfit_training_preparations(model, args):
    trainable_components = convert_to_actual_components(args.bitfit_bias_terms)
    assert args.ft_type == 'bitfit', 'ft_type should be bitfit'
    for param in model.parameters():
        param.requires_grad = False
    if trainable_components:
        trainable_components = trainable_components + ['pooler.dense.bias']
    for name, param in model.named_parameters():
        for component in trainable_components:
            if component in name:
                param.requires_grad = True
                break
    return model


def training_preparation(model, trainable_components, encoder_trainable=False):
    if encoder_trainable and trainable_components:
        raise Exception(
            f"If encoder_trainable is True, you shouldn't supply trainable_components. "
            f"Got trainable_components: {trainable_components}")



class RobertaForTokenClassification(nn.Module):

    def __init__(self, args, bert_backbone, **kwargs):
        super(RobertaForTokenClassification, self).__init__()


        self.num_labels = kwargs['num_classes']

        if args.re_init_plm:
            assert args.model_name == 'roberta-base', 'Only support Roberta-base for now'
            model_config = AutoConfig.from_pretrained(args.model_name)
            self.bert = RobertaModel(model_config)
        else:
            self.bert = AutoModel.from_pretrained(args.model_name)
            if args.ft_type == "bitfit":
                self.bert = perform_bitfit_training_preparations(self.bert, args)


        self.dropout = nn.Dropout(p=args.bert_dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)

    def forward(self, input_batch):
        input_ids, attention_mask = input_batch['input_ids'], input_batch['attention_mask']
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        # loss = None
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss()
        #     # Only keep active parts of the loss
        #     if attention_mask is not None:
        #         active_loss = attention_mask.view(-1) == 1
        #         active_logits = logits.view(-1, self.num_labels)
        #         active_labels = torch.where(
        #             active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
        #         )
        #         loss = loss_fct(active_logits, active_labels)
        #     else:
        #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # hidden_states = outputs.hidden_states,
        attentions = outputs.attentions

        return {'logits': logits, 'last_hidden_state': sequence_output, 'attentions': attentions}


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0., use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)

class ReBert(nn.Module):

    def __init__(self, args, bert_backbone, **kwargs):
        super(ReBert, self).__init__()
        self.num_labels = kwargs['num_classes']
        assert args.pooling_strategy in ['pooler_output', 'max', 'mean']
        self.pooling_strategy = args.pooling_strategy
        # self.re_init_pooler = args.re_init_pooler

        if args.re_init_plm:
            assert args.model_name == 'roberta-base', 'Only support Roberta-base for now'
            model_config = AutoConfig.from_pretrained(args.model_name)
            self.bert = RobertaModel(model_config)
        else:
            self.bert = AutoModel.from_pretrained(args.model_name)
            if args.ft_type == "bitfit":
                self.bert = perform_bitfit_training_preparations(self.bert, args)

        # self.drop = nn.Dropout(p=args.bert_dropout_rate)
        # self.out = nn.Linear(self.bert.config.hidden_size, self.num_labels)

        self.cls_fc_layer = FCLayer(self.bert.config.hidden_size, self.bert.config.hidden_size, args.bert_dropout_rate)
        self.e1_fc_layer = FCLayer(self.bert.config.hidden_size, self.bert.config.hidden_size, args.bert_dropout_rate)
        self.e2_fc_layer = FCLayer(self.bert.config.hidden_size, self.bert.config.hidden_size, args.bert_dropout_rate)
        self.label_classifier = FCLayer(self.bert.config.hidden_size * 3, self.num_labels, args.bert_dropout_rate, use_activation=False)


    def entity_average(self, hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)  # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector


    def forward(self, input_batch):
        input_ids, attention_mask, e1_mask, e2_mask = input_batch['input_ids'], \
                                                      input_batch['attention_mask'], \
                                                      input_batch['e1_mask'], \
                                                      input_batch['e2_mask']

        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = bert_out[0][:, 0, :]
        # pooler output: applies a linear layer + Tanh on the last hidden state of the [cls] token
        sequence_output = bert_out[0]
        pooled_output = bert_out['pooler_output']

        # Average
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)

        # Dropout -> tanh -> fc_layer
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.e1_fc_layer(e1_h)
        e2_h = self.e2_fc_layer(e2_h)

        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)

        logits = self.label_classifier(concat_h)

        # logits = self.out(output)
        return {'logits': logits, 'pooler_repr': pooled_output, 'cls_repr': concat_h}



# https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/
class TextBert(nn.Module):

    def __init__(self, args, bert_backbone, **kwargs):
        super(TextBert, self).__init__()
        self.num_labels = kwargs['num_classes']
        # assert args.pooling_strategy in ['pooler_output', 'max', 'mean']
        # self.pooling_strategy = args.pooling_strategy


        if args.re_init_plm:
            assert args.model_name == 'roberta-base', 'Only support Roberta-base for now'
            model_config = AutoConfig.from_pretrained(args.model_name)
            self.bert = RobertaModel(model_config)
        else:
            self.bert = AutoModel.from_pretrained(args.model_name)
            if args.ft_type == "bitfit":
                self.bert = perform_bitfit_training_preparations(self.bert, args)

        # self.bert = AutoModel.from_pretrained(args.model_name)
        self.drop = nn.Dropout(p=args.bert_dropout_rate)
        self.out = nn.Linear(self.bert.config.hidden_size, self.num_labels)


    def forward(self, input_batch):
        input_ids, attention_mask = input_batch['input_ids'], input_batch['attention_mask']
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # pooler output: applies a linear layer + Tanh on the last hidden state of the [cls] token
        cls_repr = bert_out[0][:, 0, :]

        final_repr = bert_out['pooler_output']
        output = self.drop(final_repr)
        logits = self.out(output)
        return {'logits': logits, 'cls_repr': cls_repr, 'pooler_repr': final_repr}


class AdaptiveCrossEntropy(nn.Module):
    # CrossEntropy loss that works for both text classification and named-entity recognition
    def __init__(self, args, num_classes, reduction):
        super(AdaptiveCrossEntropy, self).__init__()
        self.base_ce_fn = nn.CrossEntropyLoss(reduction=reduction)


        self.num_classes = num_classes
        if args.task_type == 'ner':
            self.loss_fn = self.ner_cross_entropy
        elif args.task_type in ['text_cls', 'text_cls_f1', 're']:
            self.loss_fn = self.txt_cls_cross_entropy
        else:
            raise ValueError("[AdaptiveCrossEntropy]: unknown task_type")

    def txt_cls_cross_entropy(self, logits, labels, attention_mask):
        return self.base_ce_fn(logits, labels)

    def ner_cross_entropy(self, logits, labels, attention_mask):
        # Only keep active parts of the loss
        loss_fct = self.base_ce_fn
        # Only keep active parts of the loss
        # Actually, I think we don't need the following if "attention_mask" condition.
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_classes)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
        else:
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        return loss

    def forward(self, logits, labels, attention_mask=None):
        return self.loss_fn(logits, labels, attention_mask)