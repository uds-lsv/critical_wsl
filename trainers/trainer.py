import torch
import os
import copy
import numpy as np
import json
from label_models.majority_voting import MajorityVotingLabelModel
import wandb
from cwsl_dataset import TextBertDataset, REBertDataset, NERBertDataset
from models import TextBert, RobertaForTokenClassification, ReBert
from adapter_models import TextBertAdapter, ReBertAdapter, RobertaForTokenClassificationAdapter
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as pr_score
from datasets import load_metric
import pandas as pd
import datetime


class Trainer:
    def __init__(self, args, logger, log_dir, random_state):
        self.args = args
        self.logger = logger
        self.log_dir = log_dir
        self.label_txt_list = None
        self.l2id = None
        self.id2l = None
        self.num_classes = None
        self.random_state = random_state
        self.store_model_flag = True if args.store_model == 1 else False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_best_val_acc_save_path = os.path.join(self.log_dir, 'best_val_acc_model.pt')
        self.best_val_optimizer_save_path = os.path.join(self.log_dir, 'best_val_acc_opt.pt')
        self.collator = None
        self.training_subset_r_state = np.random.RandomState(args.train_label_seed)
        self.validation_subset_r_state = np.random.RandomState(args.validation_label_seed)
        self.train_on_clean = True if args.train_on_clean == 1 else False
        self.validation_on_clean = True if args.validation_on_clean == 1 else False
        self.test_on_clean = True if args.test_on_clean == 1 else False
        self.data_statistics = dict()
        assert self.test_on_clean, "[WARNING]: do you really want to test on noisy test set?"

        if args.task_type == 'ner':
            self.eval_fn = self.ner_eval
            self.es_large_is_better = True  # For NER, we use validation F1-score to perform early-stopping
        elif args.task_type == 'text_cls':
            self.eval_fn = self.tc_eval
            self.es_large_is_better = False  # For TC, we use validation loss to perform early-stopping
        elif args.task_type == 'text_cls_f1':
            # Special case: we still do text-classification, but we measure F1-Score instead
            # Here, tc_eval() is used, but we use early-stopping based on F1-Score
            # Used for training on text classification with unbalanced data, e.g. the SMS dataset
            self.eval_fn = self.tc_eval
            self.es_large_is_better = True
        elif args.task_type == 're':
            self.eval_fn = self.tc_eval
            self.es_large_is_better = False
        else:
            raise ValueError("[Trainer]: unknown task_type")

    def get_batch(self, d_loader, d_iter):
        try:
            d_batch = next(d_iter)
        except StopIteration:
            d_iter = iter(d_loader)
            d_batch = next(d_iter)

        return d_batch, d_iter

    def get_val_score(self, val_score_dict):
        if self.args.task_type == 'ner':
            return val_score_dict['score_dict']['overall_f1']
        elif self.args.task_type == 'text_cls':
            return val_score_dict['loss']
        elif self.args.task_type == 'text_cls_f1':
            return val_score_dict['score_dict']['macro avg']['f1-score']
        elif self.args.task_type == 're':
            return val_score_dict['loss']
        else:
            raise ValueError("[Trainer]: unknown task_type")

    def io_id_to_bio_id(self, a):
        bio_ids = []
        last_io = -1
        for i in a:
            if i == -100:  # subtoken id, skip
                bio_ids.append(-100)
                continue
            if i == 0:
                bio_ids.append(0)
            else:
                if i == last_io:
                    bio_ids.append(int(i * 2))  # to I
                else:
                    bio_ids.append(int(i * 2 - 1))  # to B
            last_io = i
        return bio_ids

    def ner_eval(self, predictions, labels):
        now = datetime.datetime.now()
        now_str = now.strftime("%Y-%m-%d-%H-%M-%S-%f")
        metric = load_metric("seqeval", experiment_id=now_str)
        # metric = load_metric("seqeval")

        # predictions = np.argmax(predictions, axis=2)
        predictions = [self.io_id_to_bio_id(p) for p in predictions]
        labels = [self.io_id_to_bio_id(l) for l in labels]

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.id2l[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2l[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        if self.args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "overall_f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    def tc_eval(self, predictions, labels):
        classification_score_dict = classification_report(labels,
                                                          np.array(predictions).flatten(),
                                                          output_dict=True)
        return classification_score_dict

    def create_model(self, args):
        bert_config = {'num_classes': self.num_classes}
        if args.task_type == 'ner':
            if args.ft_type in ['ft', 'bitfit']:
                model = RobertaForTokenClassification(args, None, **bert_config)
            else:
                model = RobertaForTokenClassificationAdapter(args, None, **bert_config)
        elif args.task_type in ['text_cls', 'text_cls_f1']:
            if args.ft_type in ['ft', 'bitfit']:
                model = TextBert(args, None, **bert_config)
            else:
                model = TextBertAdapter(args, None, **bert_config)
        elif args.task_type == 're':
            if args.ft_type in ['ft', 'bitfit']:
                model = ReBert(args, None, **bert_config)
            else:
                model = ReBertAdapter(args, None, **bert_config)
        else:
            raise ValueError("[Model Creation]: unknown task_type")
        return model

    def compute_label_distribution(self, datasets, tags):
        # computing statistics about the clean/noisy labels, to gain a better understanding of the data
        assert len(datasets) == len(tags)
        for d, t in zip(datasets, tags):
            c_label_distribution = np.zeros(self.num_classes, dtype=np.int32)
            n_label_distribution = np.zeros(self.num_classes, dtype=np.int32)
            c_labels = copy.deepcopy(d.labels)
            n_labels = copy.deepcopy(d.n_labels)
            c_labels = np.array(c_labels)
            n_labels = np.array(n_labels)

            if self.args.task_type == 'ner':
                assert len(c_labels.shape) == len(n_labels.shape) == 2
                c_labels = c_labels.flatten()
                n_labels = n_labels.flatten()

            for cl in c_labels:
                assert cl != -1
                if self.args.task_type == 'ner' and cl == -100:
                    continue
                else:
                    c_label_distribution[cl] += 1
            for nl in n_labels:
                if self.args.task_type == 'ner' and nl == -100:
                    continue
                if nl != -1:
                    n_label_distribution[nl] += 1

            c_normalized_label_distribution = c_label_distribution / np.sum(c_label_distribution)
            n_normalized_label_distribution = n_label_distribution / np.sum(n_label_distribution)
            self.data_statistics[t] = {'c_labels': {'label_distribution': c_label_distribution.tolist(),
                                                    'normalized_label_distribution': c_normalized_label_distribution.tolist()},
                                       'n_labels': {'label_distribution': n_label_distribution.tolist(),
                                                    'normalized_label_distribution': n_normalized_label_distribution.tolist()}}

    def process_dataset(self, args, logger, log_dir, random_state, full_dataset):
        # the original dataset multiple weak labels for each instance.
        # there are different ways to aggregate the weak labels into a single label
        # the simplest way is to use the majority voting u,e, args.label_model_name == 'majority'
        # in this project, we only use the majority voting as the label aggregation method
        # but other methods can be easily added

        if args.label_model_name == 'majority':
            label_model = MajorityVotingLabelModel(args, logger, log_dir, random_state)
        else:
            raise ValueError("[Trainer]: Unknown label_model")

        weak_labels_dict = label_model.aggregate_labels(args, logger, full_dataset)
        train_weak_res, validation_weak_res, test_weak_res = weak_labels_dict["train_weak_data"], \
                                                             weak_labels_dict["validation_weak_data"], \
                                                             weak_labels_dict["test_weak_data"]

        train_set = full_dataset["train_set"]
        val_set = full_dataset["validation_set"]
        test_set = full_dataset["test_set"]
        self.l2id = full_dataset["l2id"]
        self.id2l = full_dataset["id2l"]

        if self.args.task_type == 'ner':
            num_classes = len(self.l2id.keys())
            assert num_classes % 2 != 0, "number of BIO classes should always be odd"
            self.num_classes = int((num_classes + 1) / 2)
        elif self.args.task_type in ['text_cls', 'text_cls_f1', 're']:
            self.num_classes = len(self.l2id.keys())
        else:
            raise ValueError("[Trainer]: Unknown task_type")
        n_labels = copy.deepcopy(train_weak_res['aggregated_labels'])
        train_set.n_labels = n_labels
        train_set.gen_bert_input()

        l_set, ul_set = train_set.get_covered_subset()

        val_set.n_labels = copy.deepcopy(validation_weak_res['aggregated_labels'])
        val_set.gen_bert_input()

        if not self.validation_on_clean:  # if we use a noisy validation set, we must discard the unlabeled part
            val_set_l, val_set_ul = val_set.get_covered_subset()
            val_set = val_set_l

        test_set.n_labels = copy.deepcopy(test_weak_res['aggregated_labels'])
        test_set.gen_bert_input()

        noisy_label_train_stat = self.eval_fn(l_set.n_labels, l_set.labels)
        noisy_label_test_stat = self.eval_fn(test_set.n_labels, test_set.labels)

        if self.args.task_type == 'ner':
            self.logger.info(
                f"[{self.args.label_model_name}] voting - F1 on training set: {noisy_label_train_stat['overall_f1']}")
            self.logger.info(f"[{self.args.label_model_name}] - F1 on test set: {noisy_label_test_stat['overall_f1']}")
        elif self.args.task_type == 'text_cls':  # in general, we use accuracy as the metric for text classification
            self.logger.info(
                f"[{self.args.label_model_name}] voting - Accuracy on training set: {noisy_label_train_stat['accuracy']}")
            self.logger.info(
                f"[{self.args.label_model_name}] Voting - Accuracy on test set: {noisy_label_test_stat['accuracy']}")
        elif self.args.task_type == 'text_cls_f1':  # we can sometimes use F1 as the metric for text classification if the data is very unbalanced
            self.logger.info(
                f"[{self.args.label_model_name}] voting - Macro F1 on training set: {noisy_label_train_stat['macro avg']['f1-score']}")
            self.logger.info(
                f"[{self.args.label_model_name}] voting - Macro F1 on test set: {noisy_label_test_stat['macro avg']['f1-score']}")
        elif self.args.task_type == 're':  # for the two relation extraction tasks in WRENCH, we use accuracy as the metric following previous work
            self.logger.info(
                f"[{self.args.label_model_name}] voting - Accuracy on training set: {noisy_label_train_stat['accuracy']}")
            self.logger.info(
                f"[{self.args.label_model_name}] Voting - Accuracy on test set: {noisy_label_test_stat['accuracy']}")
        else:
            raise ValueError("[Trainer]: Unknown task type")

        self.logger.info(f"weakly labeled samples: {len(l_set)}")
        self.logger.info(f"unlabeled samples: {len(ul_set)}") if ul_set is not None else None
        self.logger.info(f"validation samples: {len(val_set)}")
        self.logger.info(f"test samples: {len(test_set)}")

        return {'l_set': l_set, 'ul_set': ul_set, 'validation_set': val_set, 'test_set': test_set}

    def get_early_stopper_save_dir(self):
        if self.store_model_flag:
            early_stopper_save_dir = os.path.join(self.log_dir, 'early_stopper_model')
            if not os.path.exists(early_stopper_save_dir):
                os.makedirs(early_stopper_save_dir)
        else:
            early_stopper_save_dir = None

        return early_stopper_save_dir

    def create_subset(self, full_dataset, num_meta_samples_per_class, balanced, r_state, tag='', return_idx=False):

        if self.args.task_type != "ner":
            subset_size = num_meta_samples_per_class * self.num_classes
        else:
            subset_size = num_meta_samples_per_class

        if subset_size > len(full_dataset):
            subset_size = len(full_dataset)
        wandb.run.summary['subset_size'] = subset_size

        if balanced:
            raise NotImplementedError("[Trainer] we do not support balanced sampling for now")
        else:
            if self.args.task_type == "ner":
                rand_idx = r_state.choice(len(full_dataset), subset_size, replace=False)
            else:
                rand_idx = r_state.choice(len(full_dataset), subset_size, replace=False)

        rand_idx = list(rand_idx)

        # log into wandb table
        wandb_tag = f"subset_idx_{tag}" if tag != '' else 'subset'
        df = pd.DataFrame({wandb_tag: rand_idx})
        wandb_table = wandb.Table(data=df)
        wandb.log({wandb_tag: wandb_table})
        ri = copy.deepcopy(rand_idx)
        self.data_statistics[wandb_tag] = [int(i) for i in ri]

        # log into log directory
        log_path = os.path.join(self.log_dir, f"{wandb_tag}.csv")
        df.to_csv(log_path, index=False)

        if return_idx:
            return full_dataset.create_subset(rand_idx), rand_idx
        else:
            return full_dataset.create_subset(rand_idx)


    def save_model(self, logger, model, model_name):
        output_path = os.path.join(self.log_dir, model_name)
        torch.save(model.state_dict(), output_path)
        logger.info(f"model saved at: {output_path}")

    def train(self, args, logger, full_dataset):
        raise NotImplementedError()

    def needs_eval(self, args, global_step):
        if global_step % args.eval_freq == 0 and global_step != 0:
            return True
        else:
            return False

    def prepare_input_batch(self, args, cpu_batch, device, use_clean):
        # prepare necessary inputs for the forward pass
        input_ids = cpu_batch['input_ids'].to(device)
        attention_mask = cpu_batch['attention_mask'].to(device)

        if use_clean:
            labels = cpu_batch['c_labels'].to(device)
        else:
            labels = cpu_batch['n_labels'].to(device)

        if args.task_type == 're':
            e1_mask = cpu_batch['e1_mask'].to(device)
            e2_mask = cpu_batch['e2_mask'].to(device)
            return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 'e1_mask': e1_mask,
                    'e2_mask': e2_mask}
        else:
            return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

    def eval_model(self, args, logger, device, loss_fn,
                   eval_set_loader, model,
                   on_clean_labels, fast_mode=False, verbose=False, **kwargs):
        # perform evaluation on the eval set
        all_preds = []
        all_y = []
        model.eval()
        loss_sum = 0.0

        num_batches = len(eval_set_loader) / 10 if fast_mode else 0

        with torch.no_grad():
            for idx, eval_batch in enumerate(eval_set_loader):
                input_data = self.prepare_input_batch(args, eval_batch, device, use_clean=on_clean_labels)
                targets = input_data['labels']

                y_logits = model(input_data)['logits']
                loss = loss_fn(y_logits, targets)
                loss_sum += loss.item()
                y_preds = torch.max(y_logits, -1)[1].cpu()
                all_preds.extend(y_preds.numpy())
                all_y.extend(list(targets.cpu()))

                if fast_mode and idx > num_batches:
                    break

            score_dict = self.eval_fn(all_preds, all_y)

        return {'score_dict': score_dict,
                'loss': loss_sum / (len(all_y))}

    def log_score_to_wandb(self, args, result, global_step, tag):
        # log model performance at the current step to wandb

        if result is None:
            return

        if args.task_type == 'ner':
            wandb.log({f"{tag}/f1": result['score_dict']['overall_f1'],
                       f"{tag}/loss": result['loss']}, step=global_step)
        elif args.task_type in ['text_cls', 're']:
            wandb.log({f"{tag}/acc": result['score_dict']['accuracy'],
                       f"{tag}/macro-f1": result['score_dict']['macro avg']["f1-score"],
                       f"{tag}/loss": result['loss']}, step=global_step)
        elif args.task_type == 'text_cls_f1':
            wandb.log({f"{tag}/acc": result['score_dict']['accuracy'],
                       f"{tag}/macro-f1": result['score_dict']['macro avg']["f1-score"],
                       f"{tag}/loss": result['loss']}, step=global_step)
        else:
            raise ValueError("[Trainer]: Unknown task type")

    def summary_best_score_to_wandb(self, args, res_dict, tag):
        # log the model performance after training (as a summary) to wandb

        if args.task_type == 'ner':
            self.logger.info(f"{tag} F1: {res_dict['score_dict']['overall_f1']}")
            wandb.run.summary[f"{tag} score"] = res_dict['score_dict']['overall_f1']
        elif args.task_type == 'text_cls':
            self.logger.info(f"{tag} Accuracy: {res_dict['score_dict']['accuracy']}")
            wandb.run.summary[f"{tag} score"] = res_dict['score_dict']['accuracy']
        elif args.task_type == 'text_cls_f1':
            # raise ValueError("[Trainer]: text_cls_f1 is not implemented")
            self.logger.info(f"{tag} Macro F1: {res_dict['score_dict']['macro avg']['f1-score']}")
            wandb.run.summary[f"{tag} score"] = res_dict['score_dict']['macro avg']['f1-score']
        elif args.task_type == 're':
            self.logger.info(f"{tag} Accuracy: {res_dict['score_dict']['accuracy']}")
            wandb.run.summary[f"{tag} score"] = res_dict['score_dict']['accuracy']
        else:
            raise ValueError("[Trainer]: Unknown task type")

    def forward_backward_batch(self, args, model, data_batch, optimizer, optimizer_scheduler, ce_loss_fn,
                               use_clean_labels,
                               device):
        # https://huggingface.co/transformers/custom_datasets.html

        input_batch = self.prepare_input_batch(args,
                                               data_batch,
                                               device, use_clean=use_clean_labels)

        model.train()
        model.zero_grad()
        outputs = model(input_batch)['logits']
        loss = ce_loss_fn(outputs, input_batch['labels'])
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer_scheduler.step()  # Update learning rate schedule
        model.zero_grad()

        return {'loss': loss.item()}

    def concat_datasets(self, set1, set2):
        assert set1.id2l == set2.id2l
        if self.args.task_type == "ner":
            combined_dataset = NERBertDataset(self.args, input_data=None, tokenizer=set1.tokenizer, id2l=set1.id2l)
            combined_dataset.ids = set1.ids + set2.ids
            combined_dataset.labels = torch.cat((set1.labels, set2.labels), dim=0)
            combined_dataset.examples = set1.examples + set2.examples
            combined_dataset.weak_labels = set1.weak_labels + set2.weak_labels
            combined_dataset.n_labels = torch.cat((set1.n_labels, set2.n_labels), dim=0)
            combined_dataset.bert_input = {k: torch.cat((v, set2.bert_input[k]), dim=0) for k, v in
                                           set1.bert_input.items()}
        else:
            if self.args.task_type == "text_cls":
                combined_dataset = TextBertDataset(self.args, input_data=None, tokenizer=set1.tokenizer, id2l=set1.id2l)
            else:
                assert self.args.task_type == "re"
                combined_dataset = REBertDataset(self.args, input_data=None, tokenizer=set1.tokenizer, id2l=set1.id2l)
            combined_dataset.ids = set1.ids + set2.ids
            combined_dataset.labels = set1.labels + set2.labels
            combined_dataset.examples = set1.examples + set2.examples
            combined_dataset.weak_labels = set1.weak_labels + set2.weak_labels
            combined_dataset.n_labels = set1.n_labels + set2.n_labels
            combined_dataset.bert_input = {k: torch.cat((v, set2.bert_input[k]), dim=0) for k, v in
                                           set1.bert_input.items()}

        self.logger.info(f"Datasets concatenated, size after concatenation: {len(combined_dataset)}")

        return combined_dataset

    def extract_bert_input(self, data_batch, tag, return_idx=False):
        input_ids = data_batch[f'input_ids_{tag}']
        attention_mask = data_batch[f'attention_mask_{tag}']
        labels = data_batch[f'labels']
        if return_idx:
            return input_ids, attention_mask, labels, data_batch[f'index']
        else:
            return input_ids, attention_mask, labels

    def get_optimizer_grouped_parameters(self, args, model):
        no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.weight']
        if args.discr:
            if len(args.layer_learning_rate) > 1:
                groups = [(f'layer.{i}.', args.layer_learning_rate[i]) for i in range(12)]
            else:
                lr = args.layer_learning_rate[0]
                assert lr == args.lr
                groups = [(f'layer.{i}.', lr * pow(args.layer_learning_rate_decay, 11 - i)) for i in range(12)]
            group_all_attn_layers = [f'layer.{i}.' for i in range(12)]
            no_decay_optimizer_parameters = []
            decay_optimizer_parameters = []

            # set learning rate for self-attention layers, 12 layers for bert-base
            for g, l in groups:
                decay_optimizer_parameters.append(
                    {
                        'params': [p for n, p in model.named_parameters() if
                                   not any(nd in n for nd in no_decay) and any(nd in n for nd in [g])],
                        'weight_decay': args.weight_decay, 'lr': l
                    }
                )
                no_decay_optimizer_parameters.append(
                    {
                        'params': [p for n, p in model.named_parameters() if
                                   any(nd in n for nd in no_decay) and any(nd in n for nd in [g])],
                        'weight_decay': 0.0, 'lr': l
                    }
                )
            # set learning rate for anything that don't belong to the attention layers, e.g. embedding layer,
            # the dense layer
            group_all_parameters = [
                {'params': [p for n, p in model.named_parameters() if
                            not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all_attn_layers)],
                 'weight_decay': args.weight_decay},
                {'params': [p for n, p in model.named_parameters() if
                            any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all_attn_layers)],
                 'weight_decay': 0.0},
            ]
            optimizer_parameters = no_decay_optimizer_parameters + decay_optimizer_parameters + group_all_parameters

        else:
            optimizer_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': args.weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]

        return optimizer_parameters

    def get_trainer_info(self):
        pass