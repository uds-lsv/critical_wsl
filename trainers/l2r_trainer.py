import json
import torch
import os
from torch.utils.data import Dataset
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
import wandb
from trainers.trainer import Trainer
from trainers.early_stopper import EarlyStopper
import higher
from tqdm import tqdm
from transformers import logging as t_logging
from models import AdaptiveCrossEntropy

class LearnToReweightTrainer(Trainer):
    def __init__(self, args, logger, log_dir, random_state):
        super(LearnToReweightTrainer, self).__init__(args, logger, log_dir, random_state)
        self.l2r_meta_lr = args.l2r_meta_lr
        self.store_model_flag = True if args.store_model == 1 else False
        t_logging.set_verbosity_error()




    def train(self, args, logger, full_dataset):
        assert args.task_type in ['text_cls', 're'], "this trainer only supports text classification tasks"
        assert args.gradient_accumulation_steps <= 1, "this trainer does not support gradient accumulation for now"
        logger.info('Learn2Reweight Trainer: training started')
        device = self.device
        all_processed_data = self.process_dataset(args, logger,
                                                  self.log_dir, self.random_state, full_dataset)
        l_set, ul_set, val_set, test_set = all_processed_data['l_set'], all_processed_data['ul_set'], \
                                           all_processed_data['validation_set'], all_processed_data['test_set']
        self.compute_label_distribution(datasets=[l_set, val_set, test_set],
                                        tags=['l_set', 'val_set', 'test_set'])

        t_loader = torch.utils.data.DataLoader(test_set, batch_size=args.eval_batch_size,
                                               shuffle=True, num_workers=0)

        assert not args.small_train, "this trainer does not support small train for now"

        # save data_statistics to json
        with open(os.path.join(self.log_dir, 'data_statistics.json'), 'w') as f:
            json.dump(self.data_statistics, f)
        print(f"data statistics saved to {os.path.join(self.log_dir, 'data_statistics.json')}")

        assert val_set is not None, 'We need a validation set'

        tr_loader = torch.utils.data.DataLoader(l_set, batch_size=args.nl_batch_size,
                                                shuffle=True,
                                                num_workers=0)
        tr_iter = iter(tr_loader)
        v_loader = torch.utils.data.DataLoader(val_set,
                                               batch_size=args.eval_batch_size,
                                               shuffle=True, num_workers=0)
        val_iter = iter(v_loader)

        model = self.create_model(args)
        model = model.to(device)
        num_training_steps = args.num_training_steps
        optimizer_grouped_parameters = self.get_optimizer_grouped_parameters(args, model)

        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
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

        loss_fn_no_reduction = AdaptiveCrossEntropy(args=args, num_classes=self.num_classes, reduction='none')

        # make sure that the number of epochs covers the steps needed
        # previously there was a bug here, l_loader was written there
        num_epochs = (num_training_steps // (len(tr_loader) + 1)) + 1
        wandb.run.summary["num_epochs"] = num_epochs

        # train the network
        for step in tqdm(range(num_training_steps), desc=f'training steps', ncols=150):
            tr_batch, tr_iter = self.get_batch(tr_loader, tr_iter)
            val_batch, val_iter = self.get_batch(v_loader, val_iter)

            bs = len(tr_batch['n_labels'])

            tr_input_batch = self.prepare_input_batch(args, tr_batch, device,
                                                   use_clean=self.train_on_clean)
            val_input_batch = self.prepare_input_batch(args, val_batch, device,
                                                    use_clean=self.validation_on_clean)

            meta_net = model
            meta_optimizer = optimizer
            meta_net.zero_grad()

            with higher.innerloop_ctx(meta_net, meta_optimizer) as (fmodel, diffopt):

                # step 4, 5
                meta_pred = fmodel(tr_input_batch)['logits']
                eps = torch.zeros(bs, requires_grad=True).to(device)
                l_f_meta = torch.mean(eps * loss_fn_no_reduction(meta_pred, tr_input_batch['labels']))
                # step 7 done by higher
                diffopt.step(l_f_meta)

                meta_pred_val = fmodel(val_input_batch)['logits']
                l_g_meta = torch.mean(loss_fn_no_reduction(meta_pred_val, val_input_batch['labels']))

                grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True)[0]

            w_tilde = torch.clamp(-grad_eps.detach(), min=0)
            norm_c = torch.sum(w_tilde)

            if norm_c != 0:
                w = w_tilde / norm_c
            else:
                w = w_tilde

            model.train()
            model.zero_grad()

            outputs = model(tr_input_batch)['logits']

            loss = torch.mean(w.detach() * loss_fn_no_reduction(outputs, tr_input_batch['labels']))
            loss.backward()

            batch_ce_loss = loss.item()

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
                                          on_clean_labels=self.validation_on_clean,
                                          fast_mode=args.fast_eval,
                                          verbose=False)

                self.log_score_to_wandb(args, test_res, global_step,
                                        tag="test" if self.test_on_clean else "test_no_clean")
                self.log_score_to_wandb(args, val_res, global_step,
                                        tag="validation"if self.validation_on_clean else "validation_no_clean")

                val_score = self.get_val_score(val_res)  # track validation loss or accuracy, or F-1?
                early_stopper.register(val_score,
                                       model,
                                       optimizer, global_step)
            if global_step == num_training_steps or early_stopper.early_stop:
                break

        if self.store_model_flag:
            self.save_model(logger, model, 'last_model_weights.bin')

        test_res = self.eval_model(args, logger, device, ce_loss_fn, t_loader, model,
                                   on_clean_labels=self.test_on_clean,
                                   verbose=False)
        val_res = self.eval_model(args, logger, device, ce_loss_fn, v_loader, model,
                                  on_clean_labels=self.validation_on_clean,
                                  verbose=False)
        self.summary_best_score_to_wandb(args, test_res, tag='last')
        self.summary_best_score_to_wandb(args, val_res, tag='last-val')

        model = self.create_model(args)
        model.load_state_dict(early_stopper.get_final_res()['es_best_model'])
        model = model.to(device)
        test_res = self.eval_model(args, logger, device, ce_loss_fn, t_loader, model,
                                   on_clean_labels=self.test_on_clean,
                                   verbose=False)
        val_res = self.eval_model(args, logger, device, ce_loss_fn, v_loader, model,
                                  on_clean_labels=self.validation_on_clean,
                                  verbose=False)
        self.summary_best_score_to_wandb(args, test_res, tag='best')
        self.summary_best_score_to_wandb(args, val_res, tag='best-val')