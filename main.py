import os
import getpass
import argparse
import wandb
from pathlib import Path
from load_utils import prepare_data
from cwsl_utils import create_logger, save_args, create_trainer, save_config
import numpy as np
import torch
import random



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='trec', choices=['Yoruba', 'Hausa', 'Yoruba2', 'Hausa2',
                                                                        'trec', 'agnews', 'imdb', 'yelp', 'youtube',
                                                                        'sms',
                                                                        'spouse', 'semeval', 'chemprot',
                                                                        'mitr', 'conll', 'ontonotes'])
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--log_root', type=str, required=True)
    parser.add_argument('--exp_name', type=str, default="")
    parser.add_argument('--task_type', type=str, default='text_cls',
                        choices=['text_cls', 'ner', 'text_cls_f1', 're'])
    parser.add_argument('--trainer_name', type=str, default='vanilla',
                        choices=['vanilla',
                                 'vanilla_small_validation', 'vanilla_small_validation2', 'vanilla_small_validation3',
                                 'vanilla_small_validation_ft', 'vanilla_small_validation_ft2' ,'vanilla_small_validation_ft_ablation',
                                 'vanilla_comb', 'vanilla_comb3', 'vanilla_wc_batch', 'vanilla_wc_batch2',
                                 'vanilla_cover', 'mw',
                                 'vanilla_no_validation', 'vanilla_log', 'vanilla_on_validation',
                                 'self_train', 'self_train_no_validation',
                                 'self_train_ner', 'self_train_no_validation_ner',
                                 'cosine', 'cosine_ner', 'cosine_no_validation', 'cosine_no_validation_ner',
                                 'cosine_ner_small_validation3',
                                 'cosine_small_validation3', 'cosine_small_validation_ner',
                                 'cft', 'sft', 'lft', 'lft_ner', 'cft_ner',
                                 'l2r', 'l2r_small_validation3', 'l2r_ner_small_validation3', 'l2r_ner',
                                 'mlc',
                                 'majority', 'majority_wo', 'at_least_one', 'single_vote', 'no_conflict', 'tmwoc'])
    parser.add_argument('--label_model_name', type=str, default='majority',
                        choices=['majority', 'majority_wo', 'at_least_one', 'single_vote', 'no_conflict', 'tmwoc'])
    parser.add_argument('--model_name', type=str, default='roberta-base',
                        choices=['roberta-base', 'nyu-mll/roberta-base-10M-1', 'nyu-mll/roberta-med-small-1M-1',
                                  'bert-base-multilingual-cased'])
    parser.add_argument('--ft_type', type=str, default='ft',
                        choices=['ft', 'bitfit', 'adapter', 'adapter_lora'])
    parser.add_argument('--pooling_strategy', type=str, default='pooler_output',
                        choices=['pooler_output', 'mean', 'max'])
    parser.add_argument('--store_model', type=int, default=0, help='store model after training')
    parser.add_argument('--metric', type=str, default='accuracy', choices=['accuracy', 'f1_macro'])

    # preprocessing related
    parser.add_argument('--truncate_mode', type=str, default='last',
                        choices=['hybrid, last'], help='last: last 510 tokens, hybrid: first 128 + last 382')
    parser.add_argument('--max_sen_len', type=int, default=128)
    parser.add_argument('--special_token_offsets', type=int, default=2,
                        help='number of special tokens used in bert tokenizer for text classification')

    # BERT settings related
    parser.add_argument('--bert_dropout_rate', type=float, default=0.1)
    parser.add_argument('--re_init_plm', action='store_true')

    # Adapter LoRA related
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)

    # BitFit related
    parser.add_argument('--bitfit_bias_terms', metavar='N', type=str, nargs='+', default=['all'],
                        choices={'intermediate', 'key', 'query', 'value', 'output', 'output_layernorm',
                                 'attention_layernorm', 'all'},
                        help='bias terms to BitFit, should be given in case --fine-tune-type is bitfit '
                             '(choose \'all\' for BitFit all bias terms)')

    # training related
    parser.add_argument('--num_training_steps', type=int, default=10)
    parser.add_argument('--train_eval_freq', type=int, default=10)
    parser.add_argument('--val_eval_freq', type=int, default=50)
    parser.add_argument('--eval_freq', type=int, default=5, help='one batch is one step, '
                                                                 'eval_freq=n means eval at every n step')

    parser.add_argument('--nl_batch_size', type=int, default=16)  # this is just training batch size
    parser.add_argument('--eval_batch_size', type=int, default=50)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument('--use_clean', action='store_true',
                        help='use the ground truth label in training')
    parser.add_argument('--train_on_clean', type=int, default=0, choices=[0, 1])
    parser.add_argument('--validation_on_clean', type=int, default=1, choices=[0, 1])
    parser.add_argument('--test_on_clean', type=int, default=1, choices=[0, 1])

    parser.add_argument('--lm_ft_path', type=str, default='')

    parser.add_argument('--fast_eval', action='store_true',
                        help='use 10% of the test set for evaluation, to speed up the evaluation path')
    parser.add_argument('--freeze_bert', action='store_true',
                        help='freeze the bert backbone, i.e. use bert as feature extractor')
    parser.add_argument('--return_entity_level_metrics', action='store_true')

    # small validation/training related
    parser.add_argument('--num_samples_per_class', type=int, default=10)
    parser.add_argument('--small_train', action='store_true',
                        help='subsample the training set')
    parser.add_argument('--small_validation', action='store_true',
                        help='subsample the validation set')
    parser.add_argument('--balanced', action='store_true',
                        help='balanced sampling')
    parser.add_argument('--validation_noisy_clean_matching_ratio', type=float,
                        default=0, help='ratio of samples that have the same noisy and clean labes?')

    # optimization related
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument('--exp_decay_rate', type=float, default=0.9998)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--discr', action='store_true',
                        help='different learning rate for different layers')
    parser.add_argument('--layer_learning_rate', type=str, nargs='+', default=[2e-5])
    parser.add_argument('--layer_learning_rate_decay', type=float, default=0.95)

    # self-training related
    parser.add_argument('--teacher_init_weights_dir', type=str, default='')
    parser.add_argument('--teacher_init_type', type=str, default='ft',
                        choices=['ft', 'adapter'])
    parser.add_argument('--st_conf_threshold', type=float, default=0.9)
    parser.add_argument('--self_training_iteration', type=int, default=2)
    parser.add_argument('--student_training_steps', type=int, default=30)
    parser.add_argument('--early_stopper_delta', type=float, default=0)

    # cosine related
    parser.add_argument('--T2', type=int, default=100)
    parser.add_argument('--T3', type=int, default=50)
    parser.add_argument('--self_training_power', type=float, default=2, help='power of pred score')
    parser.add_argument('--self_training_contrastive_weight', type=float, default=1)
    parser.add_argument('--self_training_eps', type=float, default=0.1, help='confidence threshold')
    parser.add_argument('--self_training_confreg', type=float, default=0.1)
    parser.add_argument('--cosine_teacher_label_type', type=str, default='soft', choices=['soft', 'hard'])
    parser.add_argument('--cosine_distmetric', type=str, default="l2", choices=['cos', 'l2'],
                        help='distance type. Choices = [cos, l2]')
    parser.add_argument('--load_from_teacher_small_val', action='store_true',
                        help='use the small training set of the initial teacher as the validation set')

    #l2r related, l2r_meta_lr
    parser.add_argument('--l2r_meta_lr', type=float, default=0.01)

    # meta-weight related
    parser.add_argument('--mw_lr', type=float, default=0.001)

    # mlc weight related
    parser.add_argument('--mlc_lr', type=float, default=0.001)


    # test related
    parser.add_argument('--num_eval_examples', type=int, default=10000)

    # hardware related
    parser.add_argument('--wandb_offline', action='store_true',
                        help="turn off wandb sync")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--cuda_device', type=str, default="0")
    parser.add_argument('--manualSeed', type=int, default=1234, help='random seed for reproducibility')
    parser.add_argument('--train_label_seed', type=int, default=1234, help='random seed for reproducibility')
    parser.add_argument('--validation_label_seed', type=int, default=1234, help='random seed for reproducibility')

    args = parser.parse_args()

    # fix experiment seeds
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    torch.backends.cudnn.benchmark = False
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)
        torch.backends.cudnn.deterministic = True


    if args.wandb_offline:
        os.environ['WANDB_MODE'] = 'offline'

    wandb_dir = "./wandb_logs"

    Path(wandb_dir).mkdir(parents=True, exist_ok=True)

    wandb_log_name = f"critical-wsl-{args.trainer_name}" if args.exp_name=='' else args.exp_name
    wandb.init(
        project=wandb_log_name,
        dir=wandb_dir,
        config={})
    wandb.config.update(args)

    logger, log_dir = create_logger(args.log_root, args)
    save_args(log_dir, args)
    logger.info("Training started")
    print(f'log dir: {log_dir}')

    # sanity checks
    if args.dataset in ['Yoruba', 'Hausa', 'Yoruba2', 'Hausa2', 'trec', 'agnews', 'imdb', 'yelp', 'youtube']:
        assert args.task_type == 'text_cls'
    elif args.dataset in ['mitr', 'conll', 'ontonotes']:
        assert args.task_type == 'ner'
    elif args.dataset in ['sms']:
        assert args.task_type == 'text_cls_f1'
    elif args.dataset in ['spouse', 'semeval', 'chemprot']:
        assert args.task_type == 're'
    else:
        raise NotImplementedError("[main.py]: Unknown dataset type")

    if args.use_clean:
        # sanity check
        assert args.trainer_name == 'vanilla', "only vanilla trainer should be allowed to use clean data"

    logger.info(f'loading {args.dataset}')
    r_state = np.random.RandomState(args.train_label_seed)
    full_dataset = prepare_data(args, logger, r_state)

    trainer = create_trainer(args, logger, log_dir, r_state)
    trainer.train(args, logger, full_dataset)
    save_config(log_dir, 'exp_config', args)  # model_config could be updated during model creation
    logger.info(f"Logs located at {log_dir}")


if __name__=='__main__':
    main()