import argparse
import datetime
import os
import yaml
import torch

from requests.exceptions import ConnectionError, HTTPError

from huggingface_hub import login

from .evaluate_envadv import evaluate as evaluate_envadv
from .evaluate_agentadv import evaluate as evaluate_agentadv
from .evaluate_batch_agentadv import evaluate as evaluate_batch_agentadv
from .config_utils import check_evalrun_config, load_run_suffix, load_env_name
from .helpers import find_root_dir

from .access_tokens import HF_WRITE_TOKEN


def launch_evaluation(
        eval_type,
        pr_model_name, 
        pr_model_type,
        env_name,
        env_type,
        env_steps,
        eval_iters,
        eval_target,
        model_path=None,
        adv_model_name=None,
        adv_model_type=None,
        adv_model_path=None,
        omni_adv=False,
        run_suffix='',
        record_data=False,
        verbose=False,
        hf_project=None,
        device=torch.device('cpu'),
    ):
    if eval_type == 'no_adv':
        return evaluate_envadv(
            model_name=pr_model_name,
            model_type=pr_model_type,
            env_name=env_name,
            env_type=env_type,
            env_steps=env_steps,
            eval_iters=eval_iters,
            eval_target=eval_target,
            is_adv_eval=False,
            run_suffix=run_suffix,
            record_data=record_data,
            verbose=verbose,
            model_path=model_path,
            hf_project=hf_project,
            device=device,
        )
    elif eval_type == 'env_adv':
        return evaluate_envadv(
            model_name=pr_model_name,
            model_type=pr_model_type,
            env_name=env_name,
            env_type=env_type,
            env_steps=env_steps,
            eval_iters=eval_iters,
            eval_target=eval_target,
            is_adv_eval=True,
            run_suffix=run_suffix,
            record_data=record_data,
            verbose=verbose,
            model_path=model_path,
            hf_project=hf_project,
            device=device
        )
    elif eval_type == 'agent_adv':
        assert adv_model_name is not None and adv_model_type is not None, "Must provide adversarial model name and type for agent_adv evaluation."
        return evaluate_agentadv(
            pr_model_name=pr_model_name,
            pr_model_type=pr_model_type,
            pr_model_path=model_path,
            adv_model_name=adv_model_name,
            adv_model_type=adv_model_type,
            adv_model_path=adv_model_path,
            omni_adv=omni_adv,
            env_name=env_name,
            env_type=env_type,
            env_steps=env_steps,
            eval_iters=eval_iters,
            eval_target=eval_target,
            run_suffix=run_suffix,
            record_data=record_data,
            verbose=verbose,
            hf_project=hf_project,
            device=device,
        )
    elif eval_type == 'batch_noadv':
        return evaluate_batch_agentadv(
            pr_model_name=pr_model_name,
            pr_model_type=pr_model_type,
            pr_model_path=model_path,
            adv_model_name='zeroagent',
            adv_model_type='zeroagent',
            adv_model_path='evaluation_protocol/trained-models/zeroagent/model_halfcheetah.json',
            omni_adv=False,
            env_name=env_name,
            env_type=env_type,
            env_steps=env_steps,
            eval_iters=eval_iters,
            eval_target=eval_target,
            run_suffix=run_suffix,
            record_data=record_data,
            verbose=verbose,
            hf_project=hf_project,
            device=device,
        )
    elif eval_type == 'batch_agent_adv':
        assert adv_model_name is not None and adv_model_type is not None, "Must provide adversarial model name and type for batch_agent_adv evaluation."
        return evaluate_batch_agentadv(
            pr_model_name=pr_model_name,
            pr_model_type=pr_model_type,
            pr_model_path=model_path,
            adv_model_name=adv_model_name,
            adv_model_type=adv_model_type,
            adv_model_path=adv_model_path,
            omni_adv=omni_adv,
            env_name=env_name,
            env_type=env_type,
            env_steps=env_steps,
            eval_iters=eval_iters,
            eval_target=eval_target,
            run_suffix=run_suffix,
            record_data=record_data,
            verbose=verbose,
            hf_project=hf_project,
            device=device,
        )
    

if __name__ == "__main__":
    #
    # admin
    try:
        login(token=HF_WRITE_TOKEN)
    except HTTPError as e:
        print("Could not connect to HuggingFace; proceeding without, will fail if required.")
    except ConnectionError as e:
        print("Could not connect to HuggingFace; proceeding without, will fail if required.")     
    
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device == "mps":
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    # load config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, required=True, help='Name of yaml configuration file to use.')
    parser.add_argument('--omni_adv', action='store_true', help='Whether to have the adversary know the protagonist action.')
    parser.add_argument('--record_data', action='store_true', help='Whether to record evaluation data in case of two-player game.')
    parser.add_argument('--not_verbose', action='store_true', help='Whether to keep tack of evaluation procedure.')
    args = parser.parse_args()

    # load and check config
    with open(f'{find_root_dir()}/run-configs/{args.config_name}.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config = check_evalrun_config(config)

    env_name = load_env_name(config.env_type)
    run_suffix = load_run_suffix(config.run_type)

    # perform evaluation
    print(f"Starting evaluation at... {datetime.datetime.now()}")

    for model_name, model_type, model_path in zip(config.trained_model_names, config.trained_model_types, config.trained_model_paths):
        adv_model_names = config.adv_model_names if config.eval_type == 'agent_adv' or config.eval_type == 'batch_agent_adv' else [None]
        adv_model_types = config.adv_model_types if config.eval_type == 'agent_adv' or config.eval_type == 'batch_agent_adv' else [None]
        adv_model_paths = config.adv_model_paths if config.eval_type == 'agent_adv' or config.eval_type == 'batch_agent_adv' else ['']

        for adv_model_name, adv_model_type, adv_model_path in zip(adv_model_names, adv_model_types, adv_model_paths):
            # NOTE HACK HACK cannot run arrl using mps
            device = (torch.device('cpu') if device==torch.device('mps') and model_type=='arrl' or model_type=='arrl-sgld' else device)
            device = (torch.device('cpu') if device==torch.device('mps') and adv_model_type=='arrl' or model_type=='arrl-sgld' else device)

            # irrelevant loop if no explicit adversaries, otherwise runs through list of adversaries
            launch_evaluation(
                eval_type=config.eval_type,
                pr_model_name=model_name, 
                pr_model_type=model_type,
                env_name=env_name,
                env_type=config.env_type,
                env_steps=config.eval_steps,
                eval_iters=config.eval_iters,
                eval_target=config.eval_target_return,
                adv_model_name=adv_model_name,
                adv_model_type=adv_model_type,
                omni_adv=args.omni_adv,
                run_suffix=run_suffix,
                record_data=args.record_data,
                verbose=(not args.not_verbose),
                model_path=find_root_dir()[:-len('evaluation_protocol')] + model_path if model_path != 'hf' else config.hf_project + '/' + model_name,
                adv_model_path=find_root_dir()[:-len('evaluation_protocol')] + adv_model_path if adv_model_path != 'hf' else config.hf_project + '/' + adv_model_name,
                hf_project=config.hf_project,
                device=device,
            )

    print("\n========================================================================================================================")
    print(f"Done at {datetime.datetime.now()}. \n")
