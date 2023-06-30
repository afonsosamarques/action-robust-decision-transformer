import json
import os
import traceback
import datetime

import gymnasium as gym
import numpy as np
import torch
import wandb
import warnings

from collections import defaultdict

from datasets import load_dataset, load_from_disk
from huggingface_hub import login
from transformers import DecisionTransformerConfig, Trainer, TrainingArguments

from model.ardt_utils import DecisionTransformerGymDataCollator
from model.ardt_vanilla import SingleAgentRobustDT
from model.ardt_full import TwoAgentRobustDT
from model.trainable_dt import TrainableDT

from utils.helpers import set_seed_everywhere
from utils.logger import Logger

from access_tokens import HF_WRITE_TOKEN, WANDB_TOKEN


# FIXME turn into YAML
MAX_EP_LEN = 1000.0
RETURNS_SCALE = MAX_EP_LEN
MAX_RETURN = 15 * RETURNS_SCALE
BATCH_SIZE = 32
CONTEXT_SIZE = 20
N_TRAIN_STEPS = 10**6
EVAL_ITERS = 20
WANDB_PROJECT = "ARDT-Project"
SUFFIX = ''
TRACEBACK = False
IS_ADV_EVAL = True
VERBOSE_EVAL = True


def load_model(model_type, model_to_use, load_locally=False):
    prefix = f"./agents{SUFFIX}" if load_locally else "ARDT-Project"
    if model_type == "dt":
        config = DecisionTransformerConfig.from_pretrained(f"ARDT-Project/{model_to_use}", use_auth_token=True)
        model = TrainableDT(config)
        return model.from_pretrained(f"ARDT-Project/{model_to_use}", use_auth_token=True), False
    elif model_type == "ardt-vanilla":
        config = DecisionTransformerConfig.from_pretrained(f"{prefix}/{model_to_use}", use_auth_token=True)
        model = SingleAgentRobustDT(config)
        return model.from_pretrained(f"{prefix}/{model_to_use}", use_auth_token=True), True
    elif model_type == "ardt-full":
        config = DecisionTransformerConfig.from_pretrained(f"{prefix}/{model_to_use}", use_auth_token=True)
        model = TwoAgentRobustDT(config)
        return model.from_pretrained(f"{prefix}/{model_to_use}", use_auth_token=True), True
    else:
        raise Exception(f"Model {model_to_use} of type {model_type} not available.")
        

def sample_env_params(env):
    mb = env.model.body_mass
    mb = torch.tensor(mb)
    gauss = torch.distributions.Normal(mb, torch.ones_like(mb)*1.0)
    mb = gauss.sample()
    env.model.body_mass = np.array(mb)
    
    mb = env.model.opt.gravity
    mb = torch.tensor(mb)
    gauss = torch.distributions.Normal(mb, torch.ones_like(mb)*1.0)
    mb = gauss.sample()
    env.model.opt.gravity = np.array(mb)

    mb = env.model.geom_friction
    mb = torch.tensor(mb)
    gauss = torch.distributions.Normal(mb, torch.ones_like(mb)*0.1)
    mb = gauss.sample()
    env.model.geom_friction = np.array(mb)

    return env


def evaluate(model_name, model_type, device='cpu', is_adv_eval=False, verbose=False, load_locally=False):
    #
    # admin
    warnings.filterwarnings('error')
    print("==================================================")
    print(f"Evaluating model {model_name}...")
    eval_dict = {}

    # setting up environment and respective configs
    chosen_env = "HalfCheetah-v4"
    max_return = 15000.0 if chosen_env == "HalfCheetah-v4" else None
    env_target_per_1000 = 12000.0 if chosen_env == "HalfCheetah-v4" else None
    if max_return is None or env_target_per_1000 is None:
        raise Exception(f"Environment {chosen_env} not configured correctly. Missing max and target returns.")

    # load model
    model, is_adv_model = load_model(model_type, model_name, load_locally=load_locally)
    model.to(device)
    eval_dict[model_name] = defaultdict(list)

    # evaluation loop
    run_fails = 0
    run_idx = 0
    n_runs = 0

    while True:
        run_idx += 1

        if n_runs >= EVAL_ITERS:
            break

        if (run_idx+1) % 5 == 0:
            print(f"Run number {run_idx}...")
        
        try:
            with torch.no_grad():
                env = gym.make(chosen_env, render_mode="rgb_array")
                set_seed_everywhere(run_idx, env)

                if is_adv_eval:
                    env = sample_env_params(env)
                    print("Checking that sampling worked. Gravity: ", env.model.opt.gravity)
                
                state, _ = env.reset()

                returns_scale = model.config.returns_scale if "returns_scale" in model.config.to_dict().keys() else RETURNS_SCALE
                episode_return, episode_length = 0, 0
                target_return = torch.tensor(env_target_per_1000/returns_scale, device=device, dtype=torch.float32).reshape(1, 1)
                states = torch.from_numpy(state).reshape(1, model.config.state_dim).to(device=device, dtype=torch.float32)
                if is_adv_model:
                    pr_actions = torch.zeros((0, model.config.pr_act_dim), device=device, dtype=torch.float32)
                    adv_actions = torch.zeros((0, model.config.adv_act_dim), device=device, dtype=torch.float32)
                else:
                    actions = torch.zeros((0, model.config.act_dim), device=device, dtype=torch.float32)
                rewards = torch.zeros(0, device=device, dtype=torch.float32)
                timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

                for t in range(model.config.max_ep_len):
                    if is_adv_model:
                        pr_actions = torch.cat([pr_actions, torch.zeros((1, model.config.pr_act_dim), device=device)], dim=0)
                        adv_actions = torch.cat([adv_actions, torch.zeros((1, model.config.adv_act_dim), device=device)], dim=0)
                    else:
                        actions = torch.cat([actions, torch.zeros((1, model.config.act_dim), device=device)], dim=0)
                
                    rewards = torch.cat([rewards, torch.zeros(1, device=device)])

                    if is_adv_model:
                        pr_action, adv_action = model.get_action(
                            states,
                            pr_actions,
                            adv_actions,
                            rewards,
                            target_return,
                            timesteps,
                            device,
                        )
                        pr_actions[-1] = pr_action
                        adv_actions[-1] = adv_action
                        action = pr_action.detach().cpu().numpy()
                    else:
                        action = model.get_action(
                            states,
                            actions,
                            rewards,
                            target_return,
                            timesteps,
                            device,
                        )
                        actions[-1] = action
                        action = action.detach().cpu().numpy()

                    state, reward, done, _, _ = env.step(action)

                    cur_state = torch.from_numpy(state.astype(np.float32)).to(device=device).reshape(1, model.config.state_dim)
                    states = torch.cat([states, cur_state], dim=0)
                    rewards[-1] = reward

                    pred_return = target_return[0, -1] - (reward / returns_scale)
                    target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
                    timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

                    episode_return += reward
                    episode_length += 1

                    if done or t == model.config.max_ep_len - 1:
                        n_runs += 1
                        eval_dict[model_name]['iter'].append(run_idx)
                        eval_dict[model_name]['env_seed'].append(run_idx)
                        eval_dict[model_name]['init_target_return'].append(target_return[0][0].item())
                        eval_dict[model_name]['final_target_return'].append(target_return[0][-1].item())
                        eval_dict[model_name]['ep_length'].append(episode_length)
                        eval_dict[model_name]['ep_return'].append(episode_return / returns_scale)
                        run_fails = 0
                        break
        except Warning as w:
            run_fails += 1
            if run_fails > 3:
                # workaround to deal with the fact that the environment sometimes crashes
                # and I can't seem to be able to use MujocoException to catch it
                print("Too many consecutive failed runs. Stopping evaluation.")
                raise w
            print(f"Run {run_idx} failed with warning {w}")
            continue
        except Exception as e:
            run_fails += 1
            if run_fails > 3:
                # workaround to deal with the fact that the environment sometimes crashes
                # and I can't seem to be able to use MujocoException to catch it
                print("Too many consecutive failed runs. Stopping evaluation.")
                raise e
            print(f"Run {run_idx} failed with error {e}")
            continue
    
    # show some simple statistics
    if verbose:
        print("\n==========================================================================================\n")
        for model_name, metrics in eval_dict.items():
            print(f"================== {model_name} ==================")
            print(f"Initial target returns | Avg: {np.round(np.mean(metrics['init_target_return']), 4)} | Std: {np.round(np.std(metrics['init_target_return']), 4)} | Min: {np.round(np.min(metrics['init_target_return']), 4)} | Median: {np.round(np.median(metrics['init_target_return']), 4)} | Max: {np.round(np.max(metrics['init_target_return']), 4)}")
            print(f"Final target returns | Avg: {np.round(np.mean(metrics['final_target_return']), 4)} | Std: {np.round(np.std(metrics['final_target_return']), 4)} | Min: {np.round(np.min(metrics['final_target_return']), 4)} | Median: {np.round(np.median(metrics['final_target_return']), 4)} | Max: {np.round(np.max(metrics['final_target_return']), 4)}")
            print(f"Episode lengths | Avg: {np.round(np.mean(metrics['ep_length']), 4)} | Std: {np.round(np.std(metrics['ep_length']), 4)} | Min: {np.round(np.min(metrics['ep_length']), 4)} | Median: {np.round(np.median(metrics['ep_length']), 4)} | Max: {np.round(np.max(metrics['ep_length']), 4)}")
            print(f"Episode returns | Avg: {np.round(np.mean(metrics['ep_return']), 4)} | Std: {np.round(np.std(metrics['ep_return']), 4)} | Min: {np.round(np.min(metrics['ep_return']), 4)} | Median: {np.round(np.median(metrics['ep_return']), 4)} | Max: {np.round(np.max(metrics['ep_return']), 4)}")
            print("\n")

    # save eval_dict as json
    with open(f'./eval-outputs{SUFFIX}/{model_name}.json', 'w') as f:
        json.dump(eval_dict, f)

    # admin again
    warnings.resetwarnings()


if __name__ == "__main__":
    #
    # admin
    login(token=HF_WRITE_TOKEN)
    wandb.login(key=WANDB_TOKEN)
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT

    # set device
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    # FIXME turn into YAML
    # set run parameters
    datasets = ["./datasets/d4rl_expert_halfcheetah", "./datasets/rarl_halfcheetah_v1"]
    dataset_names = ["d4rl_expert", "rarl_v1"] 

    models = [SingleAgentRobustDT, TwoAgentRobustDT]
    params = [(0.1, 10.0, 10**4), (0.1, 10.0, 10**5)]
    model_names = ["ardt_vanilla_halfcheetah", "ardt_full_halfcheetah"]
    model_types = ["ardt-vanilla", "ardt-full"]

    # training/evaluation loop
    print(f"Entering loop at {datetime.datetime.now()}. Training on device {device}. \n\n")
    
    for dataset_path, dataset_name in zip(datasets, dataset_names):
        for chosen_agent, (l1, l2, warmup_steps), model_name, model_type in zip(models, params, model_names, model_types):
            dataset = load_from_disk(dataset_path)
            if dataset_name == "rarl_v1":
                # FIXME dataset is 200 trajectories longer than our standard
                dataset = dataset.select(range(200, 1200))

            print("============================================================================================================")
            print(f"\nTraining {model_name} on dataset {dataset_name} with l1={l1} and l2={l2}")

            try:
                # setting up training
                wandb.init()

                collator = DecisionTransformerGymDataCollator(
                    dataset=dataset,
                    context_size=CONTEXT_SIZE, 
                    returns_scale=RETURNS_SCALE
                )
                
                config = DecisionTransformerConfig(
                    state_dim=collator.state_dim, 
                    pr_act_dim=collator.pr_act_dim,
                    adv_act_dim=collator.adv_act_dim,
                    max_ep_len=collator.max_ep_len,
                    context_size=collator.context_size,
                    state_mean=list(collator.state_mean),
                    state_std=list(collator.state_std),
                    returns_scale=collator.returns_scale,
                    lambda1=l1,
                    lambda2=l2,
                    warmup_steps=warmup_steps,
                    max_return=MAX_RETURN
                )

                full_model_name = model_name + "-" + dataset_name
                training_args = TrainingArguments(
                    output_dir=f"./agents{SUFFIX}/" + full_model_name,
                    remove_unused_columns=False,
                    num_train_steps=10**6,
                    per_device_train_batch_size=BATCH_SIZE,
                    optim="adamw_torch",
                    learning_rate=1e-4,
                    weight_decay=1e-4,
                    warmup_steps=warmup_steps, # FIXME changed from default!!
                    max_grad_norm=0.25,
                    use_mps_device=(True if torch.backends.mps.is_available() else False),
                    push_to_hub=True,
                    dataloader_num_workers=min(4, (1 if os.cpu_count() is None else os.cpu_count) // 2),
                    log_level="info",
                    logging_steps=warmup_steps-1,
                    report_to="wandb",
                    run_name=full_model_name,
                    hub_model_id=full_model_name,
                    no_cuda=(True if torch.backends.mps.is_available() else False),
                )

                logger = Logger(
                    name=WANDB_PROJECT + "-" + full_model_name, 
                    model_name=model_name, 
                    dataset_name=dataset_name, 
                    config=config, 
                    training_args=training_args
                )

                # training
                trainer = Trainer(
                    model=chosen_agent(config, logger),
                    args=training_args,
                    train_dataset=dataset,
                    data_collator=collator,
                )

                trainer.train()
                trainer.save_model()
                logger.report_all()
                wandb.finish()

                # evaluating
                evaluate(
                    full_model_name,
                    model_type,
                    device=device,
                    is_adv_eval=IS_ADV_EVAL, 
                    verbose=VERBOSE_EVAL,
                    train_and_eval=True  # if we went through this loop, we are definitely training models to evaluate
                )
            
            except Exception as e:
                wandb.finish()
                print("====================================")
                print(f"Exception training/evaluating model {model_name}.")
                msg = "-> TRACEBACK: \n" + str(traceback.print_exc()) if TRACEBACK else "-> MESSAGE: \n" + str(e)
                print("\n\n", msg, "\n\n")

    #
    # evaluate previously-trained models
    print("============================================================================================================")
    evaluate("dt_halfcheetah-d4rl_expert", "dt", device=device, is_adv_eval=IS_ADV_EVAL, verbose=VERBOSE_EVAL)
    # evaluate("ardt_vanilla_halfcheetah-d4rl_expert", "ardt-vanilla", device=device, is_adv_eval=IS_ADV_EVAL, verbose=VERBOSE_EVAL)
    # evaluate("ardt_full_halfcheetah-d4rl_expert", "ardt-full", device=device, is_adv_eval=IS_ADV_EVAL, verbose=VERBOSE_EVAL)

    evaluate("dt_halfcheetah-rarl_v1", "dt", device=device, is_adv_eval=IS_ADV_EVAL, verbose=VERBOSE_EVAL)
    # evaluate("ardt_vanilla_halfcheetah-rarl_v1", "ardt-vanilla", device=device, is_adv_eval=IS_ADV_EVAL, verbose=VERBOSE_EVAL)
    # evaluate("ardt_full_halfcheetah-rarl_v1", "ardt-full", device=device, is_adv_eval=IS_ADV_EVAL, verbose=VERBOSE_EVAL)

    print(f"\n\nExiting at {datetime.datetime.now()}.")
