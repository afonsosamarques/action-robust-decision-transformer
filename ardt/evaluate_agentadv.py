import os
import yaml

import gymnasium as gym
import numpy as np
import torch

from requests.exceptions import HTTPError

from huggingface_hub import login

from utils.config_utils import check_evalrun_config, load_run_suffix, load_env_name, load_model
from utils.helpers import set_seed_everywhere

from access_tokens import HF_WRITE_TOKEN


def evaluate(
        model_name, 
        model_type,
        env_name,
        eval_iters,
        eval_target,
        is_adv_eval=False,
        hf_project="afonsosamarques",
        run_suffix='',
        verbose=False,
        device='cpu',
        model_path_local=None
    ):
    # load model
    try:
        model, is_adv_model = load_model(model_type, model_name, hf_project=hf_project, model_path_local=model_path_local)
    except HTTPError as e:
        print(f"Could not load model {model_name} from repo.")
    model.to(device)

    # set seed, probably needs to be set at a run-by-run level though
    set_seed_everywhere(np.random.randint(0, 10000))
    
    # evaluation loop
    print("==========================================================================================")
    print(f"Evaluating model {model_name} on environment {env_name}.")

    # TODO implement!!!!!!!!!!!!!!!!!!
    return


if __name__ == "__main__":
    #
    # admin
    login(token=HF_WRITE_TOKEN)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device == "mps":
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    # load and check config
    with open('./run-configs/evaluation.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config = check_evalrun_config(config)

    env_name = load_env_name(config.env_name)
    run_suffix = load_run_suffix(config.run_type)

    # perform evaluation
    for model_name, model_type in zip(config.trained_model_names, config.trained_model_types):
        evaluate(
            model_name=model_name, 
            model_type=model_type,
            env_name=env_name,
            eval_iters=config.eval_iters,
            eval_target=config.eval_target_return,
            is_adv_eval=config.is_adv_eval,
            hf_project=config.hf_project,
            run_suffix=run_suffix,
            verbose=True,
            device=device,
        )
