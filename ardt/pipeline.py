import argparse
import datetime
import itertools
import os
import yaml

import numpy as np
import torch
import wandb

from datasets import load_dataset, load_from_disk
from huggingface_hub import login
from transformers import DecisionTransformerConfig, Trainer, TrainingArguments

from evaluate_envadv import evaluate
from model.ardt_utils import DecisionTransformerGymDataCollator
from utils.config_utils import check_pipelinerun_config, load_run_suffix, load_env_name, load_agent, build_dataset_path, build_model_name
from utils.helpers import find_root_dir
from utils.logger import Logger

from access_tokens import HF_WRITE_TOKEN, WANDB_TOKEN


def train(
        model_name,
        chosen_agent,
        dataset_name,
        dataset,
        env_params,
        model_params,
        train_params,
        wandb_project='afonsosamarques',
        hf_project='afonsosamarques',
        run_suffix='',
        device='cpu',
    ):
    print("============================================================================================================")
    print(f"\nTraining {model_name} on dataset {dataset_name} on device {device}. Starting at {datetime.datetime.now()}.\n")
    print("================================================")

    # here we define the data treatment
    collator = DecisionTransformerGymDataCollator(
        dataset=dataset,
        context_size=model_params['context_size'],
        returns_scale=env_params['returns_scale'],
    )
    if env_params['max_ep_len'] < 0.95 * collator.max_ep_len or env_params['max_ep_len'] > 1.05 * collator.max_ep_len:
        print(f"WARNING: config max_ep_len={env_params['max_ep_len']} is not close to observed max_ep_len={collator.max_ep_len}")
    if env_params['max_ep_return'] > collator.max_ep_return:
        print(f"WARNING: config max_ep_return={env_params['max_ep_return']} is higher than observed max_ep_return={collator.max_ep_return}")
    
    # here we store both environment and model parameters
    model_config = DecisionTransformerConfig(
        state_dim=collator.state_dim, 
        pr_act_dim=collator.pr_act_dim,
        adv_act_dim=collator.adv_act_dim,
        state_mean=list(collator.state_mean),
        state_std=list(collator.state_std),
        context_size=model_params['context_size'],
        lambda1=model_params['lambda1'],
        lambda2=model_params['lambda2'],
        returns_scale=env_params['returns_scale'],
        max_ep_len=env_params['max_ep_len'],
        max_obs_len=collator.max_ep_len,
        max_ep_return=env_params['max_ep_return'],
        max_obs_return=collator.max_ep_return,
        warmup_steps=train_params['warmup_steps'],  # exception: this is used in training but due to HF API it must be in config as well
    )

    # here we define the training protocol
    training_args = TrainingArguments(
        output_dir=f"{find_root_dir()}/agents{run_suffix}/" + model_name,
        remove_unused_columns=False,
        max_steps=train_params['train_steps'],
        per_device_train_batch_size=train_params['train_batch_size'],
        optim="adamw_torch",
        learning_rate=train_params['learning_rate'],
        weight_decay=train_params['weight_decay'],
        warmup_steps=train_params['warmup_steps'],
        max_grad_norm=train_params['max_grad_norm'],
        dataloader_num_workers=min(4, (1 if os.cpu_count() is None else os.cpu_count()) // 2),
        use_mps_device=(True if torch.backends.mps.is_available() else False),
        logging_steps=train_params['warmup_steps'],
        log_level="info",
        report_to="wandb",
        run_name=model_name,
        hub_model_id=model_name,
        push_to_hub=True,
    )

    # set up and start training
    logger = Logger(
        name=wandb_project + "-" + model_name, 
        model_name=model_name, 
        model_config=model_config, 
        dataset_name=dataset_name, 
        training_args=training_args
    )

    trainer = Trainer(
        model=chosen_agent(model_config, logger),
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model()
    logger.report_all()
    wandb.finish()

    print(f"\n\nExiting at {datetime.datetime.now()}.\n")
    print("================================================\n")
    return f"{find_root_dir()}/agents{run_suffix}/" + model_name


if __name__ == "__main__":
    #
    # admin
    login(token=HF_WRITE_TOKEN)
    wandb.login(key=WANDB_TOKEN)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device == "mps":
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    # load config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, required=True, help='Name of yaml configuration file to use.')
    parser.add_argument('--is_test_run', action='store_true', help='Whether this is a test run. Set if it is, ignore if it is not.')
    args = parser.parse_args()

    with open(f'{find_root_dir()}/run-configs/{args.config_name}.yaml', 'r') as f:
        config = yaml.safe_load(f)

    config = check_pipelinerun_config(config)
    env_config = config.environment_config
    dataset_config = config.dataset_config
    model_config = config.model_config
    train_config = config.training_config
    eval_config = config.evaluation_config
    admin_config = config.admin_config

    # unwrapping some of the configs for admin purposes
    os.environ["WANDB_PROJECT"] = admin_config.wandb_project
    run_suffix = load_run_suffix(admin_config.run_type)
    
    dataset_path, dataset_name = build_dataset_path(dataset_config, env_config.env_type, is_local=dataset_config.is_local, hf_project=admin_config.hf_project)
    dataset = load_from_disk(dataset_path) if dataset_config.is_local else load_dataset(dataset_path)

    # retrieving (static) environment parameters
    env_params = {'max_ep_len': env_config.max_ep_len, 'max_ep_return': env_config.max_ep_return, 'returns_scale': env_config.returns_scale}

    # set up model/train parameter combinations
    context_size = model_config.context_size  # to iterate over
    l1 = model_config.lambda1  # to iterate over
    l2 = model_config.lambda2  # to iterate over

    train_steps = 10**train_config.train_steps
    train_batch_size = train_config.train_batch_size
    learning_rate = [10**i for i in train_config.learning_rate]  # to iterate over
    weight_decay = [10**i for i in train_config.weight_decay]  # to iterate over
    max_grad_norm = train_config.max_grad_norm  # to iterate over
    warmup_steps = [10**i for i in train_config.warmup_steps]  # to iterate over

    params = [context_size, l1, l2, learning_rate, weight_decay, max_grad_norm, warmup_steps]
    params_combinations = list(itertools.product(*params))

    # build, train and evaluate models
    for params_combination in params_combinations:
        model_params = {
            'context_size': params_combination[0],
            'lambda1': params_combination[1],
            'lambda2': params_combination[2],
        }
        train_params = {
            'train_steps': train_steps if not args.is_test_run else 10,
            'train_batch_size': train_batch_size,
            'learning_rate': params_combination[3],
            'weight_decay': params_combination[4],
            'max_grad_norm': params_combination[5],
            'warmup_steps': params_combination[6],
        }

        # set up model
        agent_type = model_config.agent_type
        env_type = env_config.env_type
        chosen_agent = load_agent(agent_type)
        model_name = build_model_name(agent_type, env_type, dataset_name)
        
        # train and recover path
        model_path_local = train(
            model_name=model_name,
            chosen_agent=chosen_agent,
            dataset_name=dataset_name,
            dataset=dataset,
            env_params=env_params,
            model_params=model_params,
            train_params=train_params,
            wandb_project=admin_config.wandb_project,
            hf_project=admin_config.hf_project,
            run_suffix=run_suffix,
            device=device,
        )

        # evaluate if desired
        if config.evaluation_config.is_eval:
            env_name = load_env_name(env_config.env_type)
            evaluate(
                model_name=model_name, 
                model_type=agent_type,
                model_path_local=model_path_local,
                env_name=env_name,
                env_type=env_config.env_type,
                eval_iters=eval_config.eval_iters if not args.is_test_run else 2,
                eval_target=eval_config.eval_target_return,
                is_adv_eval=eval_config.is_adv_eval,
                hf_project=admin_config.hf_project,
                run_suffix=run_suffix,
                verbose=admin_config.is_verbose,
                device=device,
        )

    print("\n========================================================================================================================")
    print("Done. \n")
