import os, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
from transformers import AdamW, get_linear_schedule_with_warmup
from trainers.trainer import Trainer
from trainers.early_stopper import EarlyStopper
import torch.nn.functional as F
import wandb
from models import AdaptiveCrossEntropy


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.0, metric='l2'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.metric = metric

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x0, x1, y):

        # euclidian distance
        if self.metric == 'l2':
            diff = x0 - x1
            dist_sq = torch.sum(torch.pow(diff, 2), 1) / x0.shape[-1]
            dist = torch.sqrt(dist_sq)
        elif self.metric == 'cos':
            prod = torch.sum(x0 * x1, -1)
            dist = 1 - prod / torch.sqrt(torch.sum(x0 ** 2, 1) * torch.sum(x1 ** 2, 1))
            dist_sq = dist ** 2
        else:
            print("Error Loss Metric!!")
            return 0

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss, dist_sq, dist


class CosineTrainer(Trainer):
    def __init__(self, args, logger, log_dir, random_state):
        super(CosineTrainer, self).__init__(args, logger, log_dir, random_state)

    def get_batch(self, d_loader, d_iter):
        try:
            d_batch = next(d_iter)
        except StopIteration:
            d_iter = iter(d_loader)
            d_batch = next(d_iter)

        return d_batch, d_iter

    def soft_frequency(self, logits, probs=False, soft=True):
        """
        Unsupervised Deep Embedding for Clustering Analysis
        https://arxiv.org/abs/1511.06335
        """
        power = self.args.self_training_power
        if not probs:
            softmax = nn.Softmax(dim=1)
            y = softmax(logits.view(-1, logits.shape[-1])).view(logits.shape)
        else:
            y = logits
        f = torch.sum(y, dim=0)
        t = y ** power / f
        # print('t', t)
        t = t + 1e-10
        p = t / torch.sum(t, dim=-1, keepdim=True)
        return p if soft else torch.argmax(p, dim=1)

    def calc_loss(self, input_log_prob, target, classification_loss, device, thresh=0.95, soft=True, conf='max',
                  confreg=0.1):

        if conf == 'max':
            weight = torch.max(target, dim=1).values
            w = (weight > thresh)
        elif conf == 'entropy':
            weight = torch.sum(-torch.log(target + 1e-6) * target, dim=1)  # Entropy
            weight = 1 - weight / np.log(weight.size(-1))
            w = (weight > thresh)
        else:
            raise ValueError(f'conf={conf} is unsupported')

        target = self.soft_frequency(target, probs=True, soft=soft)

        if soft:
            loss_batch = torch.sum(classification_loss(input_log_prob, target), dim=1)  # shape: bs x 1
        else:
            loss_batch = classification_loss(input_log_prob, target)

        lc = torch.mean(loss_batch[w] * weight[w])  # l_c loss in the paper

        if lc < 0:
            print("here some problems")

        n_classes_ = input_log_prob.shape[-1]
        # Note this is l-=, i.e l = l - (...)
        # below is the l_c + \lambda * R2 loss in the paper

        r2_loss_fn = nn.KLDivLoss(reduction='none')
        uniform_dist = (torch.ones(input_log_prob.shape) / n_classes_).to(device)
        r2_loss_no_reduction = r2_loss_fn(input=input_log_prob, target=uniform_dist)
        r2_loss = confreg * torch.mean(r2_loss_no_reduction[w])
        if r2_loss < 0:
            print("here some problems")

        lc_r2_sum = lc + r2_loss
        # lc -= confreg * (torch.sum(input_logits * w.unsqueeze(1)) + np.log(n_classes_) * n_classes_)

        if lc < 0:
            print("here some problems")

        res = {"lc_r2_sum": lc_r2_sum, "lc_loss": lc, "r2_loss": r2_loss, "mask": w, "weights": weight}

        return res  # the var lc here is the  'L_c + \lambda * R_2' term in the paper

    def contrastive_loss(self, input, feat, target, device, conf='none', thresh=0.1, distmetric='l2'):
        softmax = nn.Softmax(dim=1)
        target = softmax(target.view(-1, target.shape[-1])).view(target.shape)
        if conf == 'max':
            weight = torch.max(target, axis=1).values
            w = torch.tensor([i for i, x in enumerate(weight) if x > thresh], dtype=torch.long).to(device)
        elif conf == 'entropy':
            weight = torch.sum(-torch.log(target + 1e-6) * target, dim=1)
            weight = 1 - weight / np.log(weight.size(-1))
            w = torch.tensor([i for i, x in enumerate(weight) if x > thresh], dtype=torch.long).to(device)
        input_x = input[w]  # prediction with high conf

        feat_x = feat[w]
        batch_size = input_x.size()[0]
        if batch_size == 0:
            return 0
        index = torch.randperm(batch_size).to(device)
        input_y = input_x[index, :]  # permutated version of input_x
        feat_y = feat_x[index, :]
        argmax_x = torch.argmax(input_x, dim=1)
        argmax_y = torch.argmax(input_y, dim=1)
        agreement = (argmax_x == argmax_y).float()

        criterion = ContrastiveLoss(margin=1.0, metric=distmetric)
        loss, dist_sq, dist = criterion(feat_x, feat_y, agreement)

        return loss

    def train_cosine_student(self, args, logger, init_cosine_model, data_dict):
        device = self.device
        train_set = data_dict['train_set']
        v_loader = data_dict['v_loader']
        t_loader = data_dict['t_loader']
        tr_loader = torch.utils.data.DataLoader(train_set, batch_size=args.nl_batch_size, shuffle=True, num_workers=0)
        tr_iter = iter(tr_loader)

        self.logger.info(
            f"student train_set acc: {np.average(np.array(train_set.n_labels) == np.array(train_set.labels))}")
        student_model = init_cosine_model
        T2, T3 = args.T2, args.T3
        num_training_steps = T2

        optimizer_grouped_parameters = self.get_optimizer_grouped_parameters(args, student_model)
        student_optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        student_optimizer_scheduler = get_linear_schedule_with_warmup(student_optimizer,
                                                                      num_warmup_steps=args.warmup_steps,
                                                                      num_training_steps=T2)

        ce_loss_fn = nn.CrossEntropyLoss()
        eval_ce_loss_fn = AdaptiveCrossEntropy(args=args, num_classes=self.num_classes, reduction='mean')

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

        student_step = 0
        selftrain_loss = 0.0
        cosine_soft = True if args.cosine_teacher_label_type == 'soft' else False

        self_training_loss = nn.KLDivLoss(reduction='none') if cosine_soft else nn.CrossEntropyLoss(reduction='none')
        teacher_model = None
        self.logger.info(f"Total number of training steps: {num_training_steps}")
        for step in range(num_training_steps):

            if student_step % T3 == 0:
                teacher_model = copy.deepcopy(student_model)  # .to("cuda")
                teacher_model.eval()
                for p in teacher_model.parameters():
                    p.requires_grad = False

            student_model.train()
            tr_batch, tr_iter = self.get_batch(tr_loader, tr_iter)
            input_batch = self.prepare_input_batch(args, tr_batch, device, use_clean=self.train_on_clean)

            outputs = student_model(input_batch)
            outputs_pseudo = teacher_model(input_batch)
            outputs_pseudo_prob = F.softmax(outputs_pseudo['logits'], dim=1)

            logits = outputs['logits']
            log_soft_logits = torch.log(F.softmax(logits, dim=1))

            self_train_res = self.calc_loss(input_log_prob=log_soft_logits,
                                            target=outputs_pseudo_prob,
                                            classification_loss=self_training_loss,
                                            device=device,
                                            thresh=args.self_training_eps,
                                            soft=cosine_soft,
                                            conf='entropy',
                                            confreg=args.self_training_confreg)

            loss = self_train_res["lc_r2_sum"]
            wandb.log({'Train/self-train loss': loss.item()}, step=step)

            if args.self_training_contrastive_weight > 0:
                contrastive_loss = self.contrastive_loss(input=log_soft_logits,
                                                         feat=outputs_pseudo['cls_repr'],
                                                         target=outputs_pseudo['logits'],
                                                         device=device,
                                                         conf='entropy',
                                                         thresh=args.self_training_eps,
                                                         distmetric=args.cosine_distmetric)

                final_contrastive_loss = args.self_training_contrastive_weight * contrastive_loss
                wandb.log({'Train/contrastive loss': final_contrastive_loss.item()}, step=step)

                loss = loss + final_contrastive_loss

            selftrain_loss += loss
            loss.backward()
            student_optimizer.step()
            student_optimizer_scheduler.step()  # Update learning rate schedule
            student_model.zero_grad()
            teacher_model.zero_grad()
            student_step += 1

            wandb.log({f"Train/cosine loss": loss.item()}, step=student_step)

            if self.needs_eval(args, student_step):
                test_res = self.eval_model(args, logger, device, eval_ce_loss_fn, t_loader, student_model,
                                           on_clean_labels=self.test_on_clean,
                                           fast_mode=args.fast_eval,
                                           verbose=False)

                self.log_score_to_wandb(args, test_res, student_step,
                                        tag="Test(S)")

                val_res = self.eval_model(args, logger, device, eval_ce_loss_fn, v_loader, student_model,
                                          on_clean_labels=self.validation_on_clean,
                                          fast_mode=args.fast_eval,
                                          verbose=False)
                self.log_score_to_wandb(args, val_res, student_step,
                                        tag="Validation(S)")

                val_score = self.get_val_score(val_res)  # track validation loss or accuracy, or F-1?
                early_stopper.register(val_score,
                                       student_model,
                                       student_optimizer, student_step)

            if student_step == num_training_steps or early_stopper.early_stop:
                break

        student_model = self.create_model(args)
        student_weights = early_stopper.get_final_res()["es_best_model"]
        student_model.load_state_dict(student_weights)
        student_model = student_model.to(device)
        return {"global_step": student_step, "best_student_model": student_model}

    def train(self, args, logger, full_dataset):
        assert args.gradient_accumulation_steps <= 1, "this trainer does not support gradient accumulation for now"
        assert args.task_type in ['text_cls', 're'], "this cosine trainer only supports text_cls or re tasks"
        logger.info('COSINE Trainer: training started')

        all_processed_data = self.process_dataset(args, logger, self.log_dir, self.random_state, full_dataset)
        l_set, ul_set, val_set, test_set = all_processed_data['l_set'], all_processed_data['ul_set'], \
                                           all_processed_data['validation_set'], all_processed_data['test_set']

        device = self.device

        if ul_set is not None:
            train_set_full = self.concat_datasets(l_set, ul_set)
        else:
            train_set_full = l_set

        wandb.run.summary["self-train data size"] = len(train_set_full)

        t_loader = torch.utils.data.DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=True,
                                               num_workers=0)
        if args.small_validation:
            if args.load_from_teacher_small_val:
                print("Loading from the training set of the initial teacher model, and use it as the validation set")
                index_dir = '/'.join(args.teacher_init_weights_dir.split('/')[:-1])
                subset_idx_file = Path(index_dir) / "subset_idx_train.csv"
                subset_idx = pd.read_csv(subset_idx_file).values.flatten()
                val_set = copy.deepcopy(l_set)
                val_subset = val_set.create_subset(subset_idx)
                val_set = val_subset
                print(f"selected index: {subset_idx[:10]}...")
                print(f"Validation set size: {len(val_set)}")
            else:
                raise NotImplementedError("we do not support randomly sampling from validation set in COSINE yet")
        else:
            self.logger.info(f"Using full validation set")
        v_loader = torch.utils.data.DataLoader(val_set, batch_size=args.eval_batch_size,
                                               shuffle=False,
                                               num_workers=0)
        self.logger.info(f"Loading Initial Teacher from {args.teacher_init_weights_dir}")
        init_cosine_weights_dir = args.teacher_init_weights_dir
        init_cosine_weights_path = Path(init_cosine_weights_dir) / "model_dict.pt"
        init_cosine_model = self.create_model(args)
        init_cosine_model.load_state_dict(torch.load(init_cosine_weights_path))
        init_cosine_model = init_cosine_model.to(device)

        eval_ce_loss_fn = AdaptiveCrossEntropy(args=args, num_classes=self.num_classes, reduction='mean')
        initial_teacher_test_res = self.eval_model(args, logger, device, eval_ce_loss_fn,
                                                   t_loader, init_cosine_model,
                                                   on_clean_labels=self.test_on_clean,
                                                   fast_mode=False, verbose=False)

        self.summary_best_score_to_wandb(args, initial_teacher_test_res, tag='init-teacher')

        initial_teacher_val_res = self.eval_model(args, logger, device, eval_ce_loss_fn,
                                                  v_loader, init_cosine_model,
                                                  on_clean_labels=self.validation_on_clean,
                                                  fast_mode=False, verbose=False)

        self.summary_best_score_to_wandb(args, initial_teacher_val_res, tag='init-teacher-val')

        student_res = self.train_cosine_student(args, logger, init_cosine_model, {'train_set': train_set_full,
                                                                                  'v_loader': v_loader,
                                                                                  't_loader': t_loader})
        best_student_model = student_res["best_student_model"]
        test_res = self.eval_model(args, logger, device, eval_ce_loss_fn,
                                   t_loader, best_student_model,
                                   on_clean_labels=self.test_on_clean,
                                   fast_mode=args.fast_eval,
                                   verbose=False)

        v_res = self.eval_model(args, logger, device, eval_ce_loss_fn,
                                v_loader, best_student_model, fast_mode=args.fast_eval,
                                on_clean_labels=self.validation_on_clean,
                                verbose=False)
        self.summary_best_score_to_wandb(args, v_res, tag='best-s-val')
        self.summary_best_score_to_wandb(args, test_res, tag='best-s')
        self.logger.info(f"best student val score \n: {v_res['score_dict']}")
        self.logger.info(f"best student score \n: {test_res['score_dict']}")
