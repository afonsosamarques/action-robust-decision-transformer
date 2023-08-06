import argparse
import datetime
import itertools
import os
import subprocess

import numpy as np
import torch
import yaml

from collections import defaultdict
from requests.exceptions import HTTPError, ConnectionError

from datasets import Dataset
from huggingface_hub import login
from transformers import DecisionTransformerConfig, Trainer, TrainingArguments

from ardt.utils.config_utils import check_pipelinerun_config, load_run_suffix, load_env_name, build_model_name

from .discrete_models.trainable_dt import TrainableDT
from .discrete_models.ardt_simplest import SimpleRobustDT
from .discrete_models.ardt_full import TwoAgentRobustDT
from .discrete_models.ardt_utils import DecisionTransformerGymDataCollator
from .toyenv import create_onestep_toy_dataset, OneStepEnv
from .toy_advs import UniformAgent, WorstCaseAgent, ZeroAgent

from ardt.access_tokens import HF_WRITE_TOKEN, WANDB_TOKEN


############# We need local versions of these #############
def find_root_dir():
    try:
        root_dir = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode('utf-8')
    except Exception as e:
        root_dir = os.getcwd()[:os.getcwd().find('action-robust-decision-transformer')+len('action-robust-decision-transformer')]
    return root_dir + ('' if root_dir.endswith('action-robust-decision-transformer') else '/action-robust-decision-transformer') + "/codebase/toy_problem"

def load_agent(agent_type):
    if agent_type == "dt":
        return TrainableDT
    elif agent_type == "ardt-simplest" or agent_type == "ardt_simplest":
        return SimpleRobustDT
    elif agent_type == "ardt-full" or agent_type == "ardt_full":
        return TwoAgentRobustDT
    else:
        raise Exception(f"Agent type {agent_type} not available.")
    

def load_model(model_type, model_to_use, model_path):
    if model_type == "dt":
        config = DecisionTransformerConfig.from_pretrained(model_path)
        model = TrainableDT(config)
        return model.from_pretrained(model_path), False
    elif model_type == "ardt-simplest" or model_type == "ardt_simplest":
        config = DecisionTransformerConfig.from_pretrained(model_path)
        model = SimpleRobustDT(config)
        return model.from_pretrained(model_path), True
    elif model_type == "ardt-full" or model_type == "ardt_full":
        config = DecisionTransformerConfig.from_pretrained(model_path)
        model = TwoAgentRobustDT(config)
        return model.from_pretrained(model_path), True
    else:
        raise Exception(f"Model {model_to_use} of type {model_type} not available.")

############################################################


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
    num_workers = (2 if device == torch.device('mps') or device == torch.device("cpu") else min(4, os.cpu_count()-2))
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
        env_max_return = collator.max_ep_return
        print(f"WARNING: config max_ep_return={env_params['max_ep_return']} is higher than observed max_ep_return={collator.max_ep_return}. Defaulting to max episode return {env_max_return}.")
    
    # here we store both environment and model parameters
    model_config = DecisionTransformerConfig(
        state_dim=collator.state_dim, 
        act_dim=collator.pr_act_dim,
        pr_act_dim=collator.pr_act_dim ,
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
    )

    # here we define the training protocol
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
        data_seed=np.random.randint(0, 100),
        disable_tqdm=False,
        log_level="error",
        logging_strategy="no",
        save_strategy="no",
        report_to="none",
        skip_memory_metrics=True,
        run_name=model_name,
        hub_model_id=hub_model_id,
        push_to_hub=True,
    )

    # set up and start training
    trainer = Trainer(
        model=chosen_agent(model_config, None),
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model()

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

    config = check_pipelinerun_config(config, do_checks=False)
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

    # build, train and evaluate models
    dataset_name = 'toy_dataset'
    dataset = create_onestep_toy_dataset(n_trajs=train_batch_size*train_steps)  # stick the dataset in memory at once to speeds things up
    models = []

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

        models.append([model_name, agent_type, model_path_local])

    # evaluate if desired
    for model_name, agent_type, model_path_local in models:
        env_name = 'toy-env'

        for adv_model_name, adv_model_type in zip(eval_config.adv_model_names, eval_config.adv_model_types):
            # load models
            try:
                pr_model, is_adv_pr_model = load_model(agent_type, model_name, model_path=(model_path_local if model_path_local is not None else admin_config.hf_project + f"/{model_name}"))
            except HTTPError as e:
                print(f"Could not load protagonist model {model_name} from repo.")
                raise e
            pr_model = pr_model.eval(mdp_type=None)

            adv_model = None
            if adv_model_type == 'uniform':
                adv_model = UniformAgent(action_space=pr_model.model.config.pr_act_dim)
                adv_model = adv_model.eval()
            elif adv_model_type == 'worstcase':
                adv_model = WorstCaseAgent(action_space=pr_model.model.config.pr_act_dim)
                adv_model = adv_model.eval()
            elif adv_model_type == 'zero':
                adv_model = ZeroAgent(action_space=pr_model.model.config.pr_act_dim)
                adv_model = adv_model.eval()
            else:
                raise Exception(f"Adversary model {adv_model_type} not available.")

            for eval_target in [0.5, 1.5, 2.0]:
                print("\n================================================")
                print(f"Evaluating protagonist model {model_name} on environment {env_name} against adversarial model {adv_model_name} with target {eval_target}.")

                # setting up environments for parallel runs
                envs = []
                start_states = []
                for n in range(1000):
                    env = OneStepEnv()
                    state, _ = env.reset()
                    envs.append(env)
                    start_states.append(state)
                start_states = np.array(start_states)

                # evaluation loop
                episode_returns = np.zeros(1000)
                data_dict = defaultdict(list)
                for i in range(1000):
                    data_dict['observations'].append([])
                    data_dict['pr_actions'].append([])
                    data_dict['est_adv_actions'].append([])
                    data_dict['adv_actions'].append([])
                    data_dict['rewards'].append([])
                    data_dict['dones'].append([])

                with torch.no_grad():
                    pr_model.new_batch_eval(start_states=start_states, eval_target=eval_target)
                    adv_model.new_batch_eval(start_states=start_states, eval_target=eval_target)
                    
                    # run episode
                    pr_actions, est_adv_actions = pr_model.get_batch_actions(states=start_states)
                    _, adv_actions = adv_model.get_batch_actions(states=start_states, pr_actions=pr_actions)
                    cumul_actions = np.concatenate((pr_actions, adv_actions), axis=1)
                    
                    states = np.zeros_like(start_states)
                    for i, env in enumerate(envs):
                        state, reward, done, trunc, _ = env.step(cumul_actions[i])
                        states[i] = state
                        episode_returns[i] = reward

                        # log episode data
                        data_dict['observations'][i].append(start_states[i])
                        data_dict['pr_actions'][i].append(pr_actions[i])
                        data_dict['est_adv_actions'][i].append(est_adv_actions[i])
                        data_dict['adv_actions'][i].append(adv_actions[i])
                        data_dict['rewards'][i].append(0)
                        data_dict['dones'][i].append(False)
                        data_dict['observations'][i].append(state)
                        data_dict['pr_actions'][i].append(np.ones_like(pr_actions[i]) * -1.0)
                        data_dict['adv_actions'][i].append(np.ones_like(adv_actions[i]) * -1.0)
                        data_dict['est_adv_actions'][i].append(np.ones_like(est_adv_actions[i]) * -1.0)
                        data_dict['rewards'][i].append(reward)
                        data_dict['dones'][i].append(True)
            
                # show some simple statistics
                print(f"Target episode return: {eval_target}")
                print(f"Mean episode return: {np.mean(episode_returns)}")
                print(f"Std episode return: {np.std(episode_returns)}")
                print(f"Min episode return: {np.min(episode_returns)}")
                print(f"Median episode return: {np.median(episode_returns)}")
                print(f"Max episode return: {np.max(episode_returns)}")

                # count how many times each action was taken
                pr_actions = np.array(data_dict['pr_actions'])
                pr_actions = pr_actions.reshape(-1, pr_actions.shape[-1])
                possible_actions = np.unique(pr_actions, axis=0)
                possible_actions = possible_actions[np.all(possible_actions != -1.0, axis=1)]
                pr_actions_count = np.array([np.sum(np.all(pr_actions == possible_action, axis=1)) for possible_action in possible_actions])
                pr_actions_freq = pr_actions_count / (np.sum(pr_actions_count) + 1e-8)
                print(f"Chosen protagonist action statistics: ")
                for i, possible_action in enumerate(possible_actions):
                    print(f"Action {possible_action}: ({pr_actions_freq[i]}).")
                
                # count how many times each adversarial action was estimated
                est_adv_actions = np.array(data_dict['est_adv_actions'])
                est_adv_actions = est_adv_actions.reshape(-1, est_adv_actions.shape[-1])
                possible_actions = np.unique(est_adv_actions, axis=0)
                possible_actions = possible_actions[np.all(possible_actions != -1.0, axis=1)]
                est_adv_actions_count = np.array([np.sum(np.all(est_adv_actions == possible_action, axis=1)) for possible_action in possible_actions])
                est_adv_actions_freq = est_adv_actions_count / (np.sum(est_adv_actions_count) + 1e-8)
                print(f"Chosen estimated adversary action statistics: ")
                for i, possible_action in enumerate(possible_actions):
                    print(f"Action {possible_action}: ({est_adv_actions_freq[i]}).")
                if len(possible_actions) == 0: print("N/A")

                # count how many times each adversarial action was taken
                adv_actions = np.array(data_dict['adv_actions'])
                adv_actions = adv_actions.reshape(-1, adv_actions.shape[-1])
                possible_actions = np.unique(adv_actions, axis=0)
                possible_actions = possible_actions[np.all(possible_actions != -1.0, axis=1)]
                adv_actions_count = np.array([np.sum(np.all(adv_actions == possible_action, axis=1)) for possible_action in possible_actions])
                adv_actions_freq = adv_actions_count / (np.sum(adv_actions_count) + 1e-8)
                print(f"Chosen adversary action statistics: ")
                for i, possible_action in enumerate(possible_actions):
                    print(f"Action {possible_action}: ({adv_actions_freq[i]}).")

                # save data_dict as hf dataset
                data_ds = Dataset.from_dict(data_dict)
                data_ds.save_to_disk(f'{find_root_dir()}/datasets/{model_name}_{adv_model_name}_eval_{env_name}_{eval_target}')

    print("\n========================================================================================================================")
    print("Done. \n")
