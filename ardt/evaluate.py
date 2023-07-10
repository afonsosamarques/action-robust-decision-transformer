import os
import yaml

import gymnasium as gym
import numpy as np
import torch

from huggingface_hub import login

from eval_scripts import evaluate_envadv, evaluate_agentadv
from utils.config_utils import check_evalrun_config, load_run_suffix, load_env_name
from utils.helpers import find_root_dir

from access_tokens import HF_WRITE_TOKEN


def launch_evaluation(
        eval_type,
        pr_model_name, 
        pr_model_type,
        env_name,
        env_type,
        eval_iters,
        eval_target,
        adv_model_name=None,
        adv_model_type=None,
        hf_project="afonsosamarques",
        run_suffix='',
        verbose=False,
        device='cpu',
        model_path_local=None
    ):
    if eval_type == 'no_adv':
        return evaluate_envadv.evaluate(
            model_name=pr_model_name,
            model_type=pr_model_type,
            env_name=env_name,
            env_type=env_type,
            eval_iters=eval_iters,
            eval_target=eval_target,
            is_adv_eval=False,
            hf_project=hf_project,
            run_suffix=run_suffix,
            verbose=verbose,
            device=device,
            model_path_local=model_path_local
        )
    elif eval_type == 'env_adv':
        return evaluate_envadv.evaluate(
            model_name=pr_model_name,
            model_type=pr_model_type,
            env_name=env_name,
            env_type=env_type,
            eval_iters=eval_iters,
            eval_target=eval_target,
            is_adv_eval=True,
            hf_project=hf_project,
            run_suffix=run_suffix,
            verbose=verbose,
            device=device,
            model_path_local=model_path_local
        )
    elif eval_type == 'agent_adv':
        assert adv_model_name is not None and adv_model_type is not None, "Must provide adversarial model name and type for agent_adv evaluation."
        return evaluate_agentadv.evaluate(
            pr_model_name=pr_model_name,
            pr_model_type=pr_model_type,
            adv_model_name=adv_model_name,
            adv_model_type=adv_model_type,
            env_name=env_name,
            env_type=env_type,
            eval_iters=eval_iters,
            eval_target=eval_target,
            hf_project=hf_project,
            run_suffix=run_suffix,
            verbose=verbose,
            device=device,
            model_path_local=model_path_local
        )
    

if __name__ == "__main__":
    #
    # admin
    login(token=HF_WRITE_TOKEN)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device == "mps":
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    # load and check config
    with open(f'{find_root_dir()}/run-configs/evaluation.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config = check_evalrun_config(config)

    env_name = load_env_name(config.env_type)
    run_suffix = load_run_suffix(config.run_type)

    # perform evaluation
    for model_name, model_type in zip(config.trained_model_names, config.trained_model_types):
        adv_model_names = config.adv_model_names if config.eval_type == 'agent_adv' else [None]
        adv_model_types = config.adv_model_types if config.eval_type == 'agent_adv' else [None]

        for adv_model_name, adv_model_type in zip(adv_model_names, adv_model_types):
            # irrelevant loop if no explicit adversaries, otherwise runs through list of adversaries
            launch_evaluation(
                eval_type=config.eval_type,
                pr_model_name=model_name, 
                pr_model_type=model_type,
                env_name=env_name,
                env_type=config.env_type,
                eval_iters=config.eval_iters,
                eval_target=config.eval_target_return,
                adv_model_name=adv_model_name,
                adv_model_type=adv_model_type,
                hf_project=config.hf_project,
                run_suffix=run_suffix,
                verbose=True,
                device=device,
            )
