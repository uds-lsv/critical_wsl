import torch
import torch.nn as nn
from models import FCLayer
from transformers import RobertaConfig, RobertaAdapterModel, AutoAdapterModel, AutoTokenizer, AutoConfig
from transformers.adapters import LoRAConfig

class RobertaForTokenClassificationAdapter(nn.Module):

    def __init__(self, args, bert_backbone, **kwargs):
        super(RobertaForTokenClassificationAdapter, self).__init__()
        self.num_labels = kwargs['num_classes']

        assert args.model_name == 'roberta-base', 'Only support Roberta-base for now'
        config = AutoConfig.from_pretrained(args.model_name,
                                            num_label=self.num_labels)

        self.model = AutoAdapterModel.from_pretrained(args.model_name)

        if args.ft_type == 'adapter':
            self.model.add_adapter("ner")
        elif args.ft_type == 'adapter_lora':
            config = LoRAConfig(r=args.lora_r, alpha=args.lora_alpha)
            self.model.add_adapter("ner", config=config)
        else:
            raise ValueError(f"Unknown ft_type: {args.ft_type}")

        self.model.add_tagging_head("ner_head", num_labels=self.num_labels)
        print(self.model.get_labels())
        self.model.set_active_adapters([["ner"]])
        self.model.train_adapter(["ner"])

    def forward(self, input_batch):
        input_ids, attention_mask = input_batch['input_ids'], input_batch['attention_mask']
        res = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        logits = res['logits']

        return {'logits': logits}





class TextBertAdapter(nn.Module):

    def __init__(self, args, bert_backbone, **kwargs):
        super(TextBertAdapter, self).__init__()
        self.num_labels = kwargs['num_classes']

        assert args.ft_type != "ft", "ft_type should be adapter"
        assert args.model_name == "roberta-base", "Only roberta-base is supported for adapter training"
        config = RobertaConfig.from_pretrained(
            pretrained_model_name_or_path=args.model_name,
            num_labels=self.num_labels,
        )
        self.model = RobertaAdapterModel.from_pretrained(
            pretrained_model_name_or_path=args.model_name,
            config=config,
        )

        # Add a new adapter
        adapter_name = f"{args.dataset}_adapter"
        if args.ft_type == 'adapter':
            self.model.add_adapter(adapter_name=adapter_name)
        elif args.ft_type == 'adapter_lora':
            config = LoRAConfig(r=args.lora_r, alpha=args.lora_alpha)
            self.model.add_adapter(adapter_name=adapter_name, config=config)
        else:
            raise ValueError(f"Unknown ft_type: {args.ft_type}")

        # Add a matching classification head
        self.model.add_classification_head(
            head_name=adapter_name,
            num_labels=self.num_labels
        )
        self.model.train_adapter(adapter_setup=adapter_name)

    def forward(self, input_batch):
        input_ids, attention_mask = input_batch['input_ids'], input_batch['attention_mask']
        bert_out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # pooler output: applies a linear layer + Tanh on the last hidden state of the [cls] token
        cls_repr = (bert_out["hidden_states"])[-1][:, 0, :]

        logits = bert_out["logits"]  # there is no pooler_output in the adapter model
        return {'logits': logits, 'cls_repr': cls_repr}


class ReBertAdapter(nn.Module):

    def __init__(self, args, bert_backbone, **kwargs):
        super(ReBertAdapter, self).__init__()
        self.num_labels = kwargs['num_classes']

        assert args.ft_type != "ft", "ft_type should be adapter"
        assert args.model_name == "roberta-base", "Only roberta-base is supported for adapter training"
        config = RobertaConfig.from_pretrained(
            pretrained_model_name_or_path=args.model_name,
            num_labels=self.num_labels,
        )
        self.model = RobertaAdapterModel.from_pretrained(
            pretrained_model_name_or_path=args.model_name,
            config=config,
        )

        # Add a new adapter
        adapter_name = f"{args.dataset}_adapter"
        if args.ft_type == 'adapter':
            self.model.add_adapter(adapter_name=adapter_name)
        elif args.ft_type == 'adapter_lora':
            config = LoRAConfig(r=args.lora_r, alpha=args.lora_alpha)
            self.model.add_adapter(adapter_name=adapter_name, config=config)
        else:
            raise ValueError(f"Unknown ft_type: {args.ft_type}")


        self.model.train_adapter(adapter_setup=adapter_name)

        self.cls_fc_layer = FCLayer(self.model.config.hidden_size, self.model.config.hidden_size, args.bert_dropout_rate)
        self.e1_fc_layer = FCLayer(self.model.config.hidden_size, self.model.config.hidden_size, args.bert_dropout_rate)
        self.e2_fc_layer = FCLayer(self.model.config.hidden_size, self.model.config.hidden_size, args.bert_dropout_rate)
        self.label_classifier = FCLayer(self.model.config.hidden_size * 3, self.num_labels, args.bert_dropout_rate, use_activation=False)


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

        bert_out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        cls_repr = bert_out[0][:, 0, :]
        sequence_output = bert_out[0]
        pooled_output = bert_out['pooler_output']  # if we use the Adapter model, we actually don't have the pooler output

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