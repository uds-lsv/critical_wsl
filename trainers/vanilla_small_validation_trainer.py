import os
import copy
import wandb
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from models import AdaptiveCrossEntropy
from trainers.trainer import Trainer
from trainers.early_stopper import EarlyStopper
from tqdm import tqdm

class VanillaSmallValidationTrainer(Trainer):
    def __init__(self, args, logger, log_dir, random_state):
        super(VanillaSmallValidationTrainer, self).__init__(args, logger, log_dir, random_state)
        self.store_model_flag = True if args.store_model == 1 else False
        self.use_clean = args.use_clean

    def get_batch(self, d_loader, d_iter):
        try:
            d_batch = next(d_iter)
        except StopIteration:
            d_iter = iter(d_loader)
            d_batch = next(d_iter)

        return d_batch, d_iter



    def create_subset_with_new_seed(self, args, full_dataset, num_meta_samples_per_class, balanced, tag='', return_idx=False):

        if self.args.task_type != "ner":
            subset_size = num_meta_samples_per_class * self.num_classes
        else:
            subset_size = num_meta_samples_per_class

        if subset_size > len(full_dataset):
            subset_size = len(full_dataset)
        wandb.run.summary['subset_size'] = subset_size

        new_seed = args.validation_label_seed + subset_size
        r_state = np.random.RandomState(new_seed)

        wandb.run.summary['new_val_seed'] = new_seed

        if balanced:
            raise NotImplementedError("[Trainer] We need to adapt the code piece below")
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



    def train(self, args, logger, full_dataset):
        assert args.gradient_accumulation_steps <= 1, "this trainer does not support gradient accumulation for now"
        logger.info('BERT Vanilla Trainer with subsampled validation set: training started')
        device = self.device
        all_processed_data = self.process_dataset(args, logger,
                                                  self.log_dir, self.random_state, full_dataset)
        l_set, ul_set, val_set, test_set = all_processed_data['l_set'], all_processed_data['ul_set'], \
                                           all_processed_data['validation_set'], all_processed_data['test_set']
        l_loader = torch.utils.data.DataLoader(l_set, batch_size=args.nl_batch_size, shuffle=True, num_workers=0)
        t_loader = torch.utils.data.DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=True, num_workers=0)

        assert val_set is not None, 'We need a validation set'

        val_set = self.create_subset_with_new_seed(args, copy.deepcopy(val_set),
                                                   args.num_samples_per_class,
                                                   balanced=args.balanced, tag='train2val')
        print(f"working with a subset of the validation set of size {len(val_set)}")

        v_loader = torch.utils.data.DataLoader(val_set, batch_size=args.eval_batch_size,
                                               shuffle=False,
                                               num_workers=0)

        tr_loader = torch.utils.data.DataLoader(l_set, batch_size=args.nl_batch_size, shuffle=True, num_workers=0)
        tr_iter = iter(tr_loader)

        model = self.create_model(args)
        model = model.to(device)
        num_training_steps = args.num_training_steps
        optimizer_grouped_parameters = self.get_optimizer_grouped_parameters(args, model)

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        optimizer_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                              num_training_steps=num_training_steps)

        global_step = 0
        if self.store_model_flag:
            early_stopper_save_dir = os.path.join(self.log_dir, 'early_stopper_model')
            if not os.path.exists(early_stopper_save_dir):
                os.makedirs(early_stopper_save_dir)
        else:
            early_stopper_save_dir = None

        early_stopper = EarlyStopper(patience=args.patience, delta=0,
                                     large_is_better=self.es_large_is_better,
                                     save_dir=early_stopper_save_dir, verbose=False,
                                     trace_func=logger.info)

        ce_loss_fn = AdaptiveCrossEntropy(args=args, num_classes=self.num_classes, reduction='mean')

        num_epochs = (num_training_steps // (len(l_loader) + 1)) + 1
        wandb.run.summary["num_epochs"] = num_epochs

        best_val_acc = -1

        # train the network
        for step in tqdm(range(num_training_steps), desc=f'training steps', ncols=150):

            tr_batch, tr_iter = self.get_batch(tr_loader, tr_iter)
            input_batch = self.prepare_input_batch(args, tr_batch, device, use_clean=args.use_clean)

            bs = len(tr_batch['n_labels'])

            model.train()
            model.zero_grad()
            outputs = model(input_batch)['logits']
            loss = ce_loss_fn(outputs, input_batch['labels'])
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer_scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            if self.needs_eval(args, global_step):
                test_res = self.eval_model(args, logger, device, ce_loss_fn, t_loader, model,
                                           fast_mode=args.fast_eval,
                                           on_clean_labels=self.test_on_clean,
                                           verbose=False)
                val_res = self.eval_model(args, logger, device, ce_loss_fn, v_loader, model,
                                          fast_mode=args.fast_eval,
                                          on_clean_labels=self.validation_on_clean,
                                          verbose=False)

                self.log_score_to_wandb(args, test_res, global_step, tag="test")
                self.log_score_to_wandb(args, val_res, global_step, tag="validation")

                val_score = self.get_val_score(val_res)  # track validation loss or accuracy, or F-1?
                early_stopper.register(val_score,
                                       model,
                                       optimizer, global_step)
            if global_step == num_training_steps or early_stopper.early_stop:
                break

        if self.store_model_flag:
            self.save_model(logger, model, 'last_model_weights.bin')

        logger.info('BERT Vanilla Trainer with subsampled validation set: training finished. Evaluating the last-step '
                    'model...')
        test_res = self.eval_model(args, logger, device, ce_loss_fn, t_loader, model,
                                   on_clean_labels=self.test_on_clean,
                                   verbose=False)
        val_res = self.eval_model(args, logger, device, ce_loss_fn, v_loader, model,
                                  on_clean_labels=self.validation_on_clean,
                                  verbose=False)
        self.summary_best_score_to_wandb(args, test_res, tag='last')
        self.summary_best_score_to_wandb(args, val_res, tag='last-val')

        logger.info('BERT Vanilla Trainer with subsampled validation set: training finished. Loading the best model '
                    'according to early-stopping.')
        model = self.create_model(args)
        model.load_state_dict(early_stopper.get_final_res()['es_best_model'])
        model = model.to(device)
        logger.info('BERT Vanilla Trainer with subsampled validation set: training finished. Evaluating the best '
                    'model according to early-stopping.')
        test_res = self.eval_model(args, logger, device, ce_loss_fn, t_loader, model,
                                   on_clean_labels=self.test_on_clean,
                                   verbose=False)
        val_res = self.eval_model(args, logger, device, ce_loss_fn, v_loader, model,
                                  on_clean_labels=self.validation_on_clean,
                                  verbose=False)
        self.summary_best_score_to_wandb(args, test_res, tag='best')
        self.summary_best_score_to_wandb(args, val_res, tag='best-val')
