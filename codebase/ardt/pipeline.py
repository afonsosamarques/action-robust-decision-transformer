import argparse
import datetime
import itertools
import os
import subprocess

import torch
import wandb
import yaml

from requests.exceptions import HTTPError, ConnectionError

from datasets import load_dataset, load_from_disk
from huggingface_hub import login
from transformers import DecisionTransformerConfig, Trainer, TrainingArguments

from .models.model_utils import DecisionTransformerGymDataCollator
from .utils.config_utils import check_pipelinerun_config, load_run_suffix, load_env_name, load_agent, build_dataset_path, build_model_name
from .utils.helpers import find_root_dir, set_seed_everywhere
from .utils.logger import Logger

from .access_tokens import HF_WRITE_TOKEN, WANDB_TOKEN


def train(
        model_name,
        chosen_agent,
        is_multipart,
        dataset_name,
        dataset,
        env_params,
        model_params,
        train_params,
        wandb_project='afonsosamarques',
        hf_project='afonsosamarques',
        is_offline_log=False,
        run_suffix='',
        device=torch.device('cpu'),
        flag=False,
    ):
    num_workers = (0 if device == torch.device('mps') or device == torch.device("cpu") else min(4, os.cpu_count()-2))
    print("============================================================================================================")
    print(f"\nTraining {model_name} using seed {train_params['seed']} on dataset {dataset_name} on device {device} with a total of {num_workers} cores for data loading. Starting at {datetime.datetime.now()}.\n")
    print("================================================")

    # here we define the data treatment
    collator = DecisionTransformerGymDataCollator(
        dataset=dataset,
        context_size=model_params['context_size'],
        returns_scale=env_params['returns_scale'],
        is_multipart=is_multipart,
    )
    max_ep_len = collator.max_ep_len
    env_max_return = collator.max_ep_return if collator.max_ep_return % 1 == 0 else int(collator.max_ep_return) + 1

    # here we store both environment and model parameters
    model_config = DecisionTransformerConfig(
        state_dim=collator.state_dim, 
        act_dim=collator.pr_act_dim,  # hack to enable training of regular DT
        pr_act_dim=collator.pr_act_dim,
        adv_act_dim=collator.adv_act_dim,
        state_mean=list(collator.state_mean),
        state_std=list(collator.state_std),
        context_size=model_params['context_size'],
        lambda1=model_params['lambda1'],
        lambda2=model_params['lambda2'],
        returns_scale=env_params['returns_scale'],
        max_ep_len=max_ep_len,
        max_obs_len=max_ep_len,
        max_ep_return=env_max_return,
        max_obs_return=env_max_return,
        min_ep_return=collator.min_ep_return,
        min_obs_return=collator.min_ep_return,
        warmup_steps=train_params['warmup_steps'],
        total_train_steps=train_params['train_steps'],
        log_interval_steps=100,
        flag=flag,
    )

    # here we define the training protocol
    wandb_start = False
    for _ in range(10):
        # NOTE have to wait for as long as it takes, because cluster connection is a problem
        try:
            wandb.init(
                mode="offline" if is_offline_log else "online",
                project=wandb_project,
                name=model_name,
                id=model_name,
                tags=[model_name],
                dir=f"{find_root_dir()}",
                settings=wandb.Settings(_service_wait=120)
            )
            wandb_start = True
            break
        except Exception:
            continue

    hub_model_id = hf_project + "/" + model_name if hf_project is not None else model_name
    set_seed_everywhere(train_params['seed'])
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
        dataloader_num_workers=num_workers,
        data_seed=train_params['seed'],
        disable_tqdm=False,
        log_level="error",
        logging_strategy="steps",
        logging_steps=0.05,
        save_strategy="steps",
        save_steps=1/3,
        report_to=("wandb" if not is_offline_log else "none"),
        skip_memory_metrics=True,
        run_name=model_name,
        hub_model_id=hub_model_id,
        push_to_hub=(not is_offline_log),
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
    if wandb_start:
        wandb.config.update(logger.report_all(save=False))
        wandb.finish()

    print(f"\n\nExiting at {datetime.datetime.now()}.\n")
    print("================================================\n")
    return f"{find_root_dir()}/agents{run_suffix}/" + model_name


if __name__ == "__main__":
    #
    # admin
    is_offline_log = False
    try:
        login(token=HF_WRITE_TOKEN)
    except HTTPError as e:
        is_offline_log = True
        print("Could not connect to HuggingFace; proceeding without, will fail if required.")
    except ConnectionError as e:
        is_offline_log = True
        print("Could not connect to HuggingFace; proceeding without, will fail if required.")     

    try:
        wandb.login(key=WANDB_TOKEN, timeout=120)
    except Exception:
        is_offline_log = True
        print("Could not connect to wandb. Proceeding with a dry run.")

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device == torch.device("mps"):
        r = subprocess.run('export PYTORCH_ENABLE_MPS_FALLBACK=1', shell=True)
        if r.returncode != 0:
            raise RuntimeError("Could not enable MPS fallback. Exiting process.")

    # load config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, required=True, help='Name of yaml configuration file to use.')
    parser.add_argument('--flag', action='store_true', help='Flag if we want to try something different without changing the entire code.')
    args = parser.parse_args()

    with open(f'{find_root_dir()}/run-configs/{args.config_name}.yaml', 'r') as f:
        config = yaml.safe_load(f)

    config = check_pipelinerun_config(config)
    env_config = config.environment_config
    dataset_config = config.dataset_config
    model_config = config.model_config
    train_config = config.training_config
    admin_config = config.admin_config

    # unwrapping some of the configs for admin purposes
    run_suffix = load_run_suffix(admin_config.run_type)
    
    # retrieving (static) environment parameters
    env_params = {
        'env_name': load_env_name(env_config.env_type),
        'max_ep_len': env_config.max_ep_len,
        'returns_scale': env_config.returns_scale
    }

    # set up model/train parameter combinations
    context_size = model_config.context_size   # to iterate over
    if len(context_size) == 0: context_size = [20]
    l1 = model_config.lambda1  # to iterate over
    if len(l1) == 0: l1 = [1.0]
    l2 = model_config.lambda2  # to iterate over
    if len(l2) == 0: l2 = [1.0]

    train_steps = 10**train_config.train_steps
    warmup_steps = [10**i for i in train_config.warmup_steps]  # to iterate over
    if len(warmup_steps) == 0: warmup_steps = [0]
    train_batch_size = train_config.train_batch_size   # to iterate over
    if len(train_batch_size) == 0: train_batch_size = 64
    learning_rate = [10**i for i in train_config.learning_rate]  # to iterate over
    if len(learning_rate) == 0: learning_rate = [1e-4]
    weight_decay = [10**i for i in train_config.weight_decay]  # to iterate over
    if len(weight_decay) == 0: weight_decay = [1e-4]
    max_grad_norm = train_config.max_grad_norm  # to iterate over
    if len(max_grad_norm) == 0: max_grad_norm = [0.25]
    seeds = train_config.seeds  # to iterate over
    if len(seeds) == 0: seeds = [33]

    params = [context_size, l1, l2, warmup_steps, train_batch_size, learning_rate, weight_decay, max_grad_norm, seeds]
    params_combinations = list(itertools.product(*params))

    # iterate through datasets
    dataset_policies = dataset_config.online_policy_names
    dataset_types = dataset_config.dataset_types
    dataset_versions = dataset_config.dataset_versions
    dataset_is_local = dataset_config.is_local

    for dataset_policy, dataset_type, dataset_version, dataset_is_local in zip(dataset_policies, dataset_types, dataset_versions, dataset_is_local):
        dataset_path, dataset_name = build_dataset_path(
            dataset_policy,
            dataset_type,
            dataset_version, 
            env_config.env_type, 
            is_local=dataset_is_local, 
            hf_project=admin_config.hf_project
        )
        dataset = load_from_disk(dataset_path) if dataset_config.is_local else load_dataset(dataset_path, split='train')

        # build, train and evaluate models
        for params_combination in params_combinations:
            model_params = {
                'context_size': params_combination[0],
                'lambda1': params_combination[1],
                'lambda2': params_combination[2],
            }
            train_params = {
                'train_steps': train_steps,
                'warmup_steps': params_combination[3],
                'train_batch_size': params_combination[4],
                'learning_rate': params_combination[5],
                'weight_decay': params_combination[6],
                'max_grad_norm': params_combination[7],
                'seed': params_combination[8],
            }

            # set up model
            agent_type = model_config.agent_type
            env_type = env_config.env_type
            chosen_agent, is_multipart = load_agent(agent_type)
            model_name = build_model_name(agent_type, dataset_name, seed=params_combination[8])
            if is_multipart:
                model_params['context_size'] += 1
            
            # train and recover path
            model_path_local = train(
                model_name=model_name,
                chosen_agent=chosen_agent,
                is_multipart=is_multipart,
                dataset_name=dataset_name,
                dataset=dataset,
                env_params=env_params,
                model_params=model_params,
                train_params=train_params,
                wandb_project=admin_config.wandb_project,
                hf_project=admin_config.hf_project,
                is_offline_log=is_offline_log,
                run_suffix=run_suffix,
                device=device,
                flag=args.flag
            )

        print("\n========================================================================================================================")
        print("Done. \n")
