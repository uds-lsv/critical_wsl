import wandb
import torch
import copy
import os


# code based on pytorchtools -> early stopping
# https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopper:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=5, delta=0, save_dir=None, large_is_better=False,
                 verbose=False, trace_func=print, tag=''):
        """
        @param patience: How long to wait after last time validation loss improved.
        @param delta: Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        @param save_dir: Dir for the checkpoint to be saved to. Default: 'checkpoint.pt'
        @param large_is_better: Normally, we track validation loss, so smaller loss is better.
        But if you want to track something like validation accuracy, then larger is actually better.
        @param verbose:  If True, prints a message for each validation loss improvement. Default: False
        @param trace_func:
        @param tag: used to identify the early stopper, normally you don't need it.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.save_dir = save_dir
        self.trace_func = trace_func
        self.large_is_better = large_is_better
        self.tag = tag

        # buffer the best model and optimizer states
        self.best_model_state_dict = None
        self.best_optimizer_state_dict = None
        self.model_save_path = os.path.join(self.save_dir, 'model_dict.pt') if self.save_dir is not None else None
        self.optimizer_save_path = os.path.join(self.save_dir,
                                                'optimizer_dict.pt') if self.save_dir is not None else None

    def get_final_res(self):
        res = {'es_best_model': self.best_model_state_dict,
               'es_best_opt': self.best_optimizer_state_dict,
               'best_score': self.best_score}

        return res

    def register(self, current_score, model, optimizer, current_step):
        # assert not self.early_stop, "early_stop=True, you should not do more registration"
        if self.early_stop:
            self.trace_func('Actually you should stop registering scores, because early stop is already triggered')
            return

        wandb.log({f'early-stopping/validation_score{self.tag}': current_score}, step=current_step)

        if self.large_is_better:
            cur_score = -current_score
        else:
            cur_score = current_score

        if self.best_score is None:
            self.best_score = cur_score
            self.buffer_checkpoint(model, optimizer)
        elif cur_score > self.best_score - self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = cur_score
            self.buffer_checkpoint(model, optimizer)
            self.counter = 0

    def buffer_checkpoint(self, model, optimizer):
        self.best_model_state_dict = copy.deepcopy(model.state_dict())
        self.best_optimizer_state_dict = copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None

        if self.save_dir is not None:
            self.save_checkpoint(model, optimizer)
            suffix = ' and saved on disk'
        else:
            suffix = ''

        if self.verbose:
            self.trace_func(f'Best Model/Optimizer buffered{suffix}')

    def save_checkpoint(self, model, optimizer):
        """
        Saves model to disc
        """

        torch.save(model.state_dict(), self.model_save_path)
        if optimizer is not None:
            torch.save(optimizer.state_dict(), self.optimizer_save_path)
