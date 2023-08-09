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

from evaluation_protocol.evaluate import launch_evaluation
from .models.ardt_utils import DecisionTransformerGymDataCollator
from .utils.config_utils import check_pipelinerun_config, load_run_suffix, load_env_name, load_agent, build_dataset_path, build_model_name
from .utils.helpers import find_root_dir
from .utils.logger import Logger

from .access_tokens import HF_WRITE_TOKEN, WANDB_TOKEN


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
        is_offline_log=False,
        run_suffix='',
        device=torch.device('cpu'),
    ):
    num_workers = (4 if device == torch.device('mps') or device == torch.device("cpu") else min(4, os.cpu_count()-2))
    print("============================================================================================================")
    print(f"\nTraining {model_name} on dataset {dataset_name} on device {device} with a total of {num_workers} cores for data loading. Starting at {datetime.datetime.now()}.\n")
    print("================================================")

    # here we define the data treatment
    collator = DecisionTransformerGymDataCollator(
        dataset=dataset,
        context_size=model_params['context_size'],
        returns_scale=env_params['returns_scale'],
    )

    max_ep_len = env_params['max_ep_len']
    if max_ep_len < 0.95 * collator.max_ep_len or max_ep_len > 1.05 * collator.max_ep_len:
        max_ep_len = collator.max_ep_len
        print(f"WARNING: config max_ep_len={env_params['max_ep_len']} is not close to observed max_ep_len={collator.max_ep_len}. Defaulting to observed length {max_ep_len}.")
    
    env_max_return = env_params['max_ep_return']
    if env_max_return > collator.max_ep_return:
        env_max_return = collator.max_ep_return if collator.max_ep_return % 1 == 0 else round(collator.max_ep_return, 0) + 1
        print(f"WARNING: config max_ep_return={env_params['max_ep_return']} is higher than observed max_ep_return={collator.max_ep_return}. Defaulting to max episode return {env_max_return}.")
    
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
        max_ep_len=max(max_ep_len, collator.max_ep_len),
        max_obs_len=collator.max_ep_len,
        max_ep_return=max(env_max_return, collator.max_ep_return),
        max_obs_return=collator.max_ep_return,
        min_ep_return=collator.min_ep_return,
        min_obs_return=collator.min_ep_return,
        warmup_steps=train_params['warmup_steps'],  # exception: this is used in training but due to HF API it must be in config as well
        log_interval_steps=100,
        total_train_steps=train_params['train_steps']
    )

    # here we define the training protocol
    wandb.init(
        mode="offline" if is_offline_log else "online",
        project=wandb_project,
        name=model_name,
        id=model_name,
        tags=[model_name, dataset_name, env_params['env_name'], run_suffix],
        dir=f"{find_root_dir()}",
        settings=wandb.Settings(_service_wait=120)  # NOTE have to wait for as long as it takes, because cluster connection is a problem
    )

    hub_model_id = hf_project + "/" + model_name if hf_project is not None else model_name
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
        data_seed=33,
        disable_tqdm=False,
        log_level="error",
        logging_strategy="steps",
        logging_steps=0.05,
        save_strategy="steps",
        save_steps=0.2,
        report_to="wandb",
        skip_memory_metrics=True,
        run_name=model_name,
        hub_model_id=hub_model_id,
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
    run_suffix = load_run_suffix(admin_config.run_type)
    
    # retrieving (static) environment parameters
    env_params = {
        'env_name': load_env_name(env_config.env_type),
        'max_ep_len': env_config.max_ep_len, 
        'max_ep_return': env_config.max_ep_return, 
        'returns_scale': env_config.returns_scale
    }

    # set up model/train parameter combinations
    context_size = model_config.context_size  # to iterate over
    l1 = model_config.lambda1  # to iterate over
    if len(l1) == 0: l1 = [1.0]
    l2 = model_config.lambda2  # to iterate over
    if len(l2) == 0: l2 = [1.0]

    train_steps = 10**train_config.train_steps
    train_batch_size = train_config.train_batch_size
    learning_rate = [10**i for i in train_config.learning_rate]  # to iterate over
    if len(learning_rate) == 0: learning_rate = [1e-4]
    weight_decay = [10**i for i in train_config.weight_decay]  # to iterate over
    if len(weight_decay) == 0: weight_decay = [1e-4]
    max_grad_norm = train_config.max_grad_norm  # to iterate over
    if len(max_grad_norm) == 0: max_grad_norm = [0.25]
    warmup_steps = [10**i for i in train_config.warmup_steps]  # to iterate over
    if len(warmup_steps) == 0: warmup_steps = [0]

    params = [context_size, l1, l2, learning_rate, weight_decay, max_grad_norm, warmup_steps]
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
            model_name = build_model_name(agent_type, dataset_name)
            
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
                is_offline_log=is_offline_log,
                run_suffix=run_suffix,
                device=device,
            )

            # evaluate if desired
            if config.evaluation_config.is_eval:
                env_name = load_env_name(env_config.env_type)
                adv_model_names = eval_config.adv_model_names if eval_config.eval_type == 'agent_adv' else [None]
                adv_model_types = eval_config.adv_model_types if eval_config.eval_type == 'agent_adv' else [None]

                for adv_model_name, adv_model_type in zip(adv_model_names, adv_model_types):
                    # irrelevant loop if no explicit adversaries, otherwise runs through list of adversaries
                    launch_evaluation(
                        eval_type=eval_config.eval_type,
                        pr_model_name=model_name, 
                        pr_model_type=agent_type,
                        model_path_local=model_path_local,
                        env_name=env_name,
                        env_type=env_config.env_type,
                        eval_iters=eval_config.eval_iters if not args.is_test_run else 2,
                        eval_target=eval_config.eval_target_return,
                        adv_model_name=adv_model_name,
                        adv_model_type=adv_model_type,
                        hf_project=admin_config.hf_project,
                        run_suffix=run_suffix,
                        verbose=admin_config.is_verbose,
                        device=device,
                    )

        print("\n========================================================================================================================")
        print("Done. \n")
