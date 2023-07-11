import datetime

import torch

from pydantic import BaseModel, Field
from transformers import DecisionTransformerConfig

from ardt.model.ardt_simplest import SimpleRobustDT
from ardt.model.ardt_vanilla import SingleAgentRobustDT
from ardt.model.ardt_full import TwoAgentRobustDT
from ardt.model.trainable_dt import TrainableDT
from baselines.arrl.ddpg import DDPG


############################ Common ############################
def load_run_suffix(run_type):
    if run_type == "core":
        return ''
    elif run_type == "pipeline":
        return '-pipeline'
    elif run_type == "test":
        return '-test'
    else:
        raise Exception(f"Run type {run_type} not available.")
    

def load_env_name(env_type):
    if env_type == "halfcheetah":
        return "HalfCheetah-v4"
    elif env_type == "hopper":
        return "Hopper-v4"
    elif env_type == "walker2d":
        return "Walker2d-v4"
    else:
        raise Exception(f"Environment {env_type} not available.")


def load_ddpg_model(path):
    var_dict = torch.load(f"{path}/ddpg_vars")
    agent = DDPG(gamma=var_dict['gamma'], tau=var_dict['tau'], hidden_size=var_dict['hidden_size'], num_inputs=var_dict['num_inputs'],
                 action_space=var_dict['action_space'], train_mode=False, alpha=0, replay_size=0, normalize_obs=True)

    agent.actor.load_state_dict(torch.load(f"{path}/ddpg_actor", map_location=lambda storage, loc: storage))
    agent.adversary.load_state_dict(torch.load(f"{path}/ddpg_adversary", map_location=lambda storage, loc: storage))

    var_dict = torch.load(f"{path}/ddpg_vars")
    if var_dict['obs_rms_mean'] is not None:
        agent.obs_rms.mean = var_dict['obs_rms_mean']
        agent.obs_rms.var = var_dict['obs_rms_var']
        agent.normalize_observations = True
    else:
        agent.obs_rms = None
        agent.normalize_observations = False

    return agent, True


def load_model(model_type, model_to_use, model_path):
    if model_type == "dt":
        config = DecisionTransformerConfig.from_pretrained(model_path, use_auth_token=True)
        model = TrainableDT(config)
        return model.from_pretrained(model_path, use_auth_token=True), False
    elif model_type == "ardt-simplest" or model_type == "ardt_simplest":
        config = DecisionTransformerConfig.from_pretrained(model_path, use_auth_token=True)
        model = SimpleRobustDT(config)
        return model.from_pretrained(model_path, use_auth_token=True), True
    elif model_type == "ardt-vanilla" or model_type == "ardt_vanilla":
        config = DecisionTransformerConfig.from_pretrained(model_path, use_auth_token=True)
        model = SingleAgentRobustDT(config)
        return model.from_pretrained(model_path, use_auth_token=True), True
    elif model_type == "ardt-full" or model_type == "ardt_full":
        config = DecisionTransformerConfig.from_pretrained(model_path, use_auth_token=True)
        model = TwoAgentRobustDT(config)
        return model.from_pretrained(model_path, use_auth_token=True), True
    elif model_type == "arrl":
        return load_ddpg_model(model_path), True
    else:
        raise Exception(f"Model {model_to_use} of type {model_type} not available.")


def build_model_name(model_type, env_type, dataset_name):
    datetime_encoding = datetime.datetime.now().strftime("%d%m_%H%M")
    return f"{model_type}-{env_type}-{dataset_name}-{datetime_encoding}"



############################ Evaluation Script ############################
class EvaluationRunConfig(BaseModel):
    env_type: str = Field(...)
    run_type: str = Field(...)
    hf_project: str = Field(...)
    trained_model_names: list[str] = Field(...)
    trained_model_types: list[str] = Field(...)
    trained_model_paths: list[str] = Field(...)
    adv_model_names: list[str] = Field(...)
    adv_model_types: list[str] = Field(...)
    eval_type: str = Field(...)
    eval_target_return: int = Field(...)
    eval_iters: int = Field(...)
    eval_steps: int = Field(...)


def check_evalrun_config(config):
    config = EvaluationRunConfig(**config)
    assert config.eval_type in ['no_adv', 'env_adv', 'agent_adv'], "Evaluation type needs to be either 'no_adv', 'env_adv' or 'agent_adv'."
    assert len(config.trained_model_names) == len(config.trained_model_types), "There need to be as many model names as model types."
    if config.eval_type == 'agent_adv':
        assert len(config.adv_model_names) > 0, "There need to be at least one adversarial model."
        assert len(config.adv_model_names) == len(config.adv_model_types), "There need to be as many adversarial model names as adversarial model types."
    assert all([mt in ['dt', 'ardt-simplest', 'ardt_simplest', 'ardt-vanilla', 'ardt_vanilla', 'ardt-full', 'ardt_full', 'arrl'] for mt in config.trained_model_types]), "Model types need to be either 'dt', 'ardt-simplest', 'ardt-vanilla', 'ardt-full' or 'arrl'."
    assert config.run_type in ['core', 'pipeline', 'test'], "Run type needs to be either 'core', 'pipeline' or 'test'."
    assert config.env_type in ['halfcheetah', 'hopper', 'walker2d'], "Environment name needs to be either 'halfcheetah', 'hopper' or 'walker2d'."
    return config
