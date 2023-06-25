import json
import gymnasium as gym
import numpy as np
import torch

from collections import defaultdict

from datasets import load_dataset, load_from_disk
from huggingface_hub import login
from transformers import DecisionTransformerModel, DecisionTransformerConfig, Trainer, TrainingArguments

from model.ardt_utils import DecisionTransformerGymDataCollator
from model.ardt_vanilla import SingleAgentRobustDT
from model.ardt_full import TwoAgentRobustDT

from utils.helpers import set_seed_everywhere

from access_tokens import HF_WRITE_TOKEN


RETURNS_SCALE = 1000.0
CONTEXT_SIZE = 20
N_EPOCHS = 300 
WARMUP_EPOCHS = 30
EVAL_ITERS = 20


def get_action_no_adv(model, states, actions, rewards, returns_to_go, timesteps, device):
    # NOTE this implementation does not condition on past rewards
    # reshape to model input format
    states = states.reshape(1, -1, model.config.state_dim)
    actions = actions.reshape(1, -1, model.config.act_dim)
    returns_to_go = returns_to_go.reshape(1, -1, 1)
    timesteps = timesteps.reshape(1, -1)

    # normalisation constants
    state_mean = torch.from_numpy(np.array(model.config.state_mean).astype(np.float32)).to(device=device)
    state_std = torch.from_numpy(np.array(model.config.state_std).astype(np.float32)).to(device=device)

    # retrieve window of observations based on context length
    states = states[:, -model.config.context_size :]
    actions = actions[:, -model.config.context_size :]
    returns_to_go = returns_to_go[:, -model.config.context_size :]
    timesteps = timesteps[:, -model.config.context_size :]

    # normalisation
    states = (states - state_mean) / state_std

    # pad all tokens to sequence length
    padlen = model.config.context_size - states.shape[1]
    attention_mask = torch.cat([torch.zeros(padlen, device=device), torch.ones(states.shape[1], device=device)]).to(dtype=torch.long).reshape(1, -1)
    states = torch.cat([torch.zeros((1, padlen, model.config.state_dim), device=device), states], dim=1).float()
    actions = torch.cat([torch.zeros((1, padlen, model.config.act_dim), device=device), actions], dim=1).float()
    returns_to_go = torch.cat([torch.zeros((1, padlen, 1), device=device), returns_to_go], dim=1).float()
    timesteps = torch.cat([torch.zeros((1, padlen), dtype=torch.long, device=device), timesteps], dim=1)

    # forward pass
    _, action_preds, _ = model.forward(
        states=states,
        actions=actions,
        rewards=rewards,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
        return_dict=False,
    )

    return action_preds[0, -1]


def load_model(model_type, model_to_use):
    if model_type == "dt":
        return DecisionTransformerModel.from_pretrained(f"afonsosamarques/{model_to_use}", use_auth_token=True), False
    elif model_type == "ardt-vanilla":
        config = DecisionTransformerConfig.from_pretrained(f"afonsosamarques/{model_to_use}", use_auth_token=True)
        model = SingleAgentRobustDT(config)
        return model.from_pretrained(f"afonsosamarques/{model_to_use}", use_auth_token=True), True
    elif model_type == "ardt-full":
        config = DecisionTransformerConfig.from_pretrained(f"afonsosamarques/{model_to_use}", use_auth_token=True)
        model = TwoAgentRobustDT(config)
        return model.from_pretrained(f"afonsosamarques/{model_to_use}", use_auth_token=True), True
    else:
        raise Exception(f"Model {model_to_use} of type {model_type} not available.")
        

def evaluate(model_name, model_type):
    chosen_env = "HalfCheetah-v4"
    env_target_per_1000 = 12000
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    eval_dict = {}
    print("==================================================")
    print(f"Evaluating model {model_name}...")
    model, is_adv = load_model(model_type, model_name)
    model.to(device)
    eval_dict[model_name] = defaultdict(list)

    for i in range(EVAL_ITERS):
        if (i+1) % 5 == 0:
            print(f"Run number {i}...")
        
        with torch.no_grad():
            env = gym.make(chosen_env, render_mode="rgb_array")
            set_seed_everywhere(i*3, env)
            state, _ = env.reset()

            returns_scale = model.config.returns_scale if "returns_scale" in model.config.to_dict().keys() else 1000.0  # NOTE compatibility
            episode_return, episode_length = 0, 0
            target_return = torch.tensor(env_target_per_1000/returns_scale, device=device, dtype=torch.float32).reshape(1, 1)
            states = torch.from_numpy(state).reshape(1, model.config.state_dim).to(device=device, dtype=torch.float32)
            if is_adv:
                pr_actions = torch.zeros((0, model.config.pr_act_dim), device=device, dtype=torch.float32)
                adv_actions = torch.zeros((0, model.config.adv_act_dim), device=device, dtype=torch.float32)
            else:
                actions = torch.zeros((0, model.config.act_dim), device=device, dtype=torch.float32)
            rewards = torch.zeros(0, device=device, dtype=torch.float32)
            timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

            for t in range(model.config.max_ep_len):
                if is_adv:
                    pr_actions = torch.cat([pr_actions, torch.zeros((1, model.config.pr_act_dim), device=device)], dim=0)
                    adv_actions = torch.cat([adv_actions, torch.zeros((1, model.config.adv_act_dim), device=device)], dim=0)
                else:
                    actions = torch.cat([actions, torch.zeros((1, model.config.act_dim), device=device)], dim=0)
            
                rewards = torch.cat([rewards, torch.zeros(1, device=device)])

                if is_adv:
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
                    action = get_action_no_adv(
                        model,
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
                    eval_dict[model_name]['iter'].append(i)
                    eval_dict[model_name]['env_seed'].append(i*3)
                    eval_dict[model_name]['init_target_return'].append(target_return[0][0].item())
                    eval_dict[model_name]['final_target_return'].append(target_return[0][-1].item())
                    eval_dict[model_name]['ep_length'].append(episode_length)
                    eval_dict[model_name]['ep_return'].append(episode_return / returns_scale)
                    break

    # show some simple statistics
    print("\n==========================================================================================\n")
    for model_name, metrics in eval_dict.items():
        print(f"================== {model_name} ==================")
        print(f"Initial target returns | Avg: {np.round(np.mean(metrics['init_target_return']), 4)} | Std: {np.round(np.std(metrics['init_target_return']), 4)} | Min: {np.round(np.min(metrics['init_target_return']), 4)} | Median: {np.round(np.median(metrics['init_target_return']), 4)} | Max: {np.round(np.max(metrics['init_target_return']), 4)}")
        print(f"Final target returns | Avg: {np.round(np.mean(metrics['final_target_return']), 4)} | Std: {np.round(np.std(metrics['final_target_return']), 4)} | Min: {np.round(np.min(metrics['final_target_return']), 4)} | Median: {np.round(np.median(metrics['final_target_return']), 4)} | Max: {np.round(np.max(metrics['final_target_return']), 4)}")
        print(f"Episode lengths | Avg: {np.round(np.mean(metrics['ep_length']), 4)} | Std: {np.round(np.std(metrics['ep_length']), 4)} | Min: {np.round(np.min(metrics['ep_length']), 4)} | Median: {np.round(np.median(metrics['ep_length']), 4)} | Max: {np.round(np.max(metrics['ep_length']), 4)}")
        print(f"Episode returns | Avg: {np.round(np.mean(metrics['ep_return']), 4)} | Std: {np.round(np.std(metrics['ep_return']), 4)} | Min: {np.round(np.min(metrics['ep_return']), 4)} | Median: {np.round(np.median(metrics['ep_return']), 4)} | Max: {np.round(np.max(metrics['ep_return']), 4)}")
        print("\n")

    # save eval_dict as json
    with open(f'./eval-outputs/{model_name}.json', 'w') as f:
        json.dump(eval_dict, f)


if __name__ == "__main__":
    login(token=HF_WRITE_TOKEN)

    datasets = ["./datasets/d4rl_expert_halfcheetah", "./datasets/rarl_halfcheetah_v1"]
    dataset_names = ["d4rl", "rarl"] 

    models = [SingleAgentRobustDT, SingleAgentRobustDT, SingleAgentRobustDT, TwoAgentRobustDT, TwoAgentRobustDT, TwoAgentRobustDT]

    params = [(0, 0), 
              (0.05, 0),
              (0.05, 8.0),
              (0, 0), 
              (0.05, 0),
              (0.05, 8.0)]

    model_names = ["ardt_vanilla_pr_no_entropy", 
                   "ardt_vanilla_pr_all", 
                   "ardt_vanilla_all_plus_adv",
                   "ardt_full_pr_no_entropy", 
                   "ardt_full_pr_all", 
                   "ardt_full_all_plus_adv"]
    
    model_types = ["ardt-vanilla",
                   "ardt-vanilla",
                   "ardt-vanilla",
                   "ardt-full",
                   "ardt-full",
                   "ardt-full"]

    for dataset_path, dataset_name in zip(datasets, dataset_names):
        for chosen_agent, (l1, l2), model_name, model_type in zip(models, params, model_names, model_types):
            dataset = load_from_disk(dataset_path)
            if dataset_name == "rarl":
                dataset = dataset.select(range(200, 1200))

            print("============================================================================================================")
            print(f"\nTraining {chosen_agent} with l1={l1} and l2={l2}")

            try:
                collator = DecisionTransformerGymDataCollator(dataset, context_size=CONTEXT_SIZE, returns_scale=RETURNS_SCALE)
                config = DecisionTransformerConfig(state_dim=collator.state_dim, 
                                                pr_act_dim=collator.pr_act_dim,
                                                adv_act_dim=collator.adv_act_dim,
                                                max_ep_len=collator.max_ep_len,
                                                context_size=collator.context_size,
                                                state_mean=list(collator.state_mean),
                                                state_std=list(collator.state_std),
                                                scale=collator.scale,
                                                lambda1=l1,
                                                lambda2=l2,
                                                warmup_epochs=WARMUP_EPOCHS,
                                                returns_scale=RETURNS_SCALE,
                                                max_return=15000) # NOTE completely random, potentially not needed
                model = chosen_agent(config)

                full_model_name = model_name + "_" + dataset_name
                training_args = TrainingArguments(
                    output_dir="./agents-pipeline/" + full_model_name,
                    remove_unused_columns=False,
                    num_train_epochs=N_EPOCHS,
                    per_device_train_batch_size=64,
                    optim="adamw_torch",
                    learning_rate=1e-4,
                    weight_decay=1e-4,
                    warmup_ratio=0.1,
                    max_grad_norm=0.25,
                    use_mps_device=True,
                    push_to_hub=True,
                    report_to="none",
                    hub_model_id=full_model_name,
                )

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=dataset,
                    data_collator=collator,
                )

                trainer.train()
                trainer.save_model()
                evaluate(full_model_name, model_type)
            except Exception as e:
                print("====================================")
                print(f"Exception training/evaluating model {chosen_agent}. Skipping...")
                print(e)

    print("============================================================================================================")
    evaluate("dt-halfcheetah-v2", "dt")
