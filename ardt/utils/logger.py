import json
import random

import numpy as np
import torch
import wandb


class Logger:
    def __init__(self, name, model_name, dataset_name, config, training_args):
        self.name = name
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.config = config
        self.training_args = training_args
        self.entries = []

    def add_entry(self, step, hyperparams, tr_losses, dist_params={}, eval_losses={}, log=False):
        def _log_entry(step, hyperparams, dist_params, tr_losses, eval_losses=None):
            entry = {}
            entry['step'] = step
            entry['hyperparams'] = hyperparams
            entry['dist_params'] = dist_params
            entry['tr_losses'] = tr_losses
            if eval_losses is not None:
                entry['eval_losses'] = eval_losses
            return entry

        def _log_wandb(self, step, entry):
            for key, value in entry.entry.items():
                if key != 'step':
                    wandb.log({key: value}, step=step)

        log_entry = _log_entry(
            step=step,
            hyperparams=hyperparams,
            dist_params=dist_params,
            tr_losses=tr_losses,
            eval_losses=eval_losses
        )
        self.entries.append(log_entry)
        if log:
            assert step is not None
            _log_wandb(step, log_entry)

    def report_all(self, with_entries=False):
        report = {
            "python_rnd": random.getstate(),
            "np_rnd": np.random.get_state(),
            "pytorch_rnd": torch.get_rng_state(),
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "max_episode_length": self.config.max_ep_len,
            "context_window_size": self.config.context_size,
            "returns_scale": self.config.returns_scale,
            "max_return": self.config.max_return,
            "init_lambda1": self.config.lambda1,
            "init_lambda2": self.config.lambda2,
            "warmup_epochs": self.config.warmup_epochs,
            "e2e_epochs": self.training_args.num_train_epochs - self.config.warmup_epochs,
            "batch_size": self.training_args.per_device_train_batch_size,
            "init_learning_rate": self.training_args.learning_rate,
            "weight_decay": self.training_args.weight_decay,
            "adam_beta1": self.training_args.adam_beta1,
            "adam_beta2": self.training_args.adam_beta2,
            "adam_epsilon": self.training_args.adam_epsilon,
            "warmup_ratio": self.training_args.warmup_ratio,
            "max_grad_norm": self.training_args.max_grad_norm,
        }

        if with_entries:
            report['entries'] = self.entries

        with open(f'./wandb-json/{self.name}.json', 'w') as f:
            json.dump(report, f, indent=4)

        return report
