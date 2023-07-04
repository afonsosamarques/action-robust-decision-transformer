import json
import random

import numpy as np
import torch
import wandb


class Logger:
    def __init__(self, name, model_name, model_config, dataset_name, training_args):
        self.name = name
        self.model_name = model_name
        self.model_config = model_config
        self.dataset_name = dataset_name
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

        def _log_wandb(step, entry):
            for key, value in entry.items():
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
            # model/env config
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "returns_scale": self.model_config.returns_scale,
            "max_episode_length": self.model_config.max_ep_len,
            "max_obs_length": self.model_config.max_obs_len,
            "max_episode_return": self.model_config.max_ep_return,
            "max_obs_return": self.model_config.max_obs_return,
            # model config
            "context_window_size": self.model_config.context_size,
            "init_lambda1": self.model_config.lambda1,
            "init_lambda2": self.model_config.lambda2,
            # train config
            "total_steps": self.training_args.num_train_epochs,
            "warmup_steps": self.training_args.warmup_steps,
            "batch_size": self.training_args.per_device_train_batch_size,
            "init_learning_rate": self.training_args.learning_rate,
            "weight_decay": self.training_args.weight_decay,
            "adam_beta1": self.training_args.adam_beta1,
            "adam_beta2": self.training_args.adam_beta2,
            "adam_epsilon": self.training_args.adam_epsilon,
            "warmup_steps": self.training_args.warmup_steps,
            "max_grad_norm": self.training_args.max_grad_norm,
            "dropout": self.training_args.dropout,
        }

        if with_entries:
            report['entries'] = self.entries

        with open(f'./wandb-json/{self.name}.json', 'w') as f:
            json.dump(report, f, indent=4)

        return report
