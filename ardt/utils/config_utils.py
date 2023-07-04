import hashlib
import os
import subprocess

from pydantic import BaseModel, Field
from transformers import DecisionTransformerConfig

from model.ardt_vanilla import SingleAgentRobustDT
from model.ardt_full import TwoAgentRobustDT
from model.trainable_dt import TrainableDT


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
    

def load_env_name(env_name):
    if env_name == "halfcheetah":
        return "HalfCheetah-v4"
    elif env_name == "hopper":
        return "Hopper-v4"
    elif env_name == "walker2d":
        return "Walker2d-v4"
    else:
        raise Exception(f"Environment {env_name} not available.")
    

############################ Evaluation Script ############################
class EvaluationRunConfig(BaseModel):
    env_name: str = Field(...)
    run_type: str = Field(...)
    trained_model_names: list[str] = Field(...)
    trained_model_types: list[str] = Field(...)
    is_adv_eval: bool = Field(...)
    eval_target_return: int = Field(...)
    eval_iters: int = Field(...)
    hf_project: str = Field(...)


def check_evalrun_config(config):
    config = EvaluationRunConfig(**config)
    assert len(config.trained_model_names) == len(config.trained_model_types), "There need to be as many model names as model types."
    assert all([model_type in ['dt', 'ardt-vanilla', 'ardt-full'] for model_type in config.trained_model_types]), "Model types need to be either 'dt' or 'ardt-vanilla' or 'ardt-full."
    assert config.run_type in ['core', 'pipeline', 'test'], "Run type needs to be either 'core', 'pipeline' or 'test'."
    assert config.env_name in ['halfcheetah', 'hopper', 'walker2d'], "Environment name needs to be either 'halfcheetah', 'hopper' or 'walker2d'."
    assert config.hf_project in ['afonsosamarques', 'ARDT-Project'], "HF project needs to be either 'afonsosamarques' or 'ARDT-Project'."
    return config


def load_model(model_type, model_to_use, hf_project="afonsosamarques", model_path_local=None):
    model_path = model_path_local if model_path_local is not None else f"{hf_project}/{model_to_use}"
    if model_type == "dt":
        config = DecisionTransformerConfig.from_pretrained(model_path, use_auth_token=True)
        model = TrainableDT(config)
        return model.from_pretrained(model_path, use_auth_token=True), False
    elif model_type == "ardt-vanilla":
        config = DecisionTransformerConfig.from_pretrained(model_path, use_auth_token=True)
        model = SingleAgentRobustDT(config)
        return model.from_pretrained(model_path, use_auth_token=True), True
    elif model_type == "ardt-full":
        config = DecisionTransformerConfig.from_pretrained(model_path, use_auth_token=True)
        model = TwoAgentRobustDT(config)
        return model.from_pretrained(model_path, use_auth_token=True), True
    else:
        raise Exception(f"Model {model_to_use} of type {model_type} not available.")


############################ Pipeline Script ############################
class EnvironmentConfig(BaseModel):
    env_name: str = Field(...)
    max_ep_len: int = Field(...)
    returns_scale: int = Field(...)
    max_ep_return: int = Field(...)


class DatasetConfig(BaseModel):
    online_policy_name : str = Field(...)
    dataset_type: str = Field(...)
    dataset_version: str = Field(...)


class ModelConfig(BaseModel):
    model_type: str = Field(...)
    context_size: list[int] = Field(...)
    lambda1: list[float] = Field(...)
    lambda2: list[float] = Field(...)


class TrainingConfig(BaseModel):
    train_steps: int = Field(...)
    train_batch_size: int = Field(...)
    learning_rate: list[float] = Field(...)
    weight_decay: list[float] = Field(...)
    max_grad_norm: list[float] = Field(...)
    warmup_steps: list[int] = Field(...)
    dropout: list[float] = Field(...)


class EvaluationConfig(BaseModel):
    is_eval: bool = Field(...)
    is_adv_eval: bool = Field(...)
    eval_target_return: int = Field(...)
    eval_iters: int = Field(...)


class AdminConfig(BaseModel):
    wandb_project: str = Field(...)
    hf_project: str = Field(...)
    run_type: str = Field(...)
    is_verbose: bool = Field(...)
    print_tracebacks: bool = Field(...)


class PipelineConfig(BaseModel):
    environment_config: EnvironmentConfig = Field(...)
    dataset_config: DatasetConfig = Field(...)
    model_config: ModelConfig = Field(...)
    training_config: TrainingConfig = Field(...)
    evaluation_config: EvaluationConfig = Field(...)
    admin_config: AdminConfig = Field(...)


def check_pipelinerun_config(config):
    config = PipelineConfig(**config)
    assert config.environment_config.env_name in ['halfcheetah', 'hopper', 'walker2d'], "Environment name needs to be either 'halfcheetah', 'hopper' or 'walker2d'."
    assert config.dataset_config.online_policy_name in ['d4rl', 'rarl'], "Online policy needs to be either 'd4rl' or 'rarl'."
    assert config.dataset_config.dataset_type in ['train', 'test', 'expert'], "Dataset type needs to be either 'train' or 'test' or 'expert'."
    assert config.model_config.model_type in ['dt', 'ardt-vanilla', 'ardt-full']
    assert config.admin_config.wandb_project in ['afonsosamarques', 'ARDT-Project'], "Wandb project needs to be either 'afonsosamarques' or 'ARDT-Project'."
    assert config.admin_config.hf_project in ['afonsosamarques', 'ARDT-Project'], "Wandb project needs to be either 'afonsosamarques' or 'ARDT-Project'."
    assert config.admin_config.run_type in ['core', 'pipeline', 'test'], "Run type needs to be either 'core', 'pipeline' or 'test'."
    return config


def load_agent(model_type):
    if model_type == "dt":
        return TrainableDT
    elif model_type == "ardt-vanilla" or model_type == "ardt_vanilla":
        return SingleAgentRobustDT
    elif model_type == "ardt-full" or model_type == "ardt_full":
        return TwoAgentRobustDT
    else:
        raise Exception(f"Model type {model_type} not available.")


def find_root_dir():
    try:
        root_dir = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode('utf-8')
    except Exception as e:
        root_dir = os.getcwd()[:os.getcwd().find('action-robust-decision-transformer')+len('action-robust-decision-transformer')]
    return root_dir


def build_dataset_path(dataset_config, env_name):
    dataset_dir = find_root_dir() + "/datasets"
    dataset_path = f"{dataset_dir}/{dataset_config.online_policy_name}_{dataset_config.dataset_type}_{env_name}"
    dataset_path += f"_{dataset_config.dataset_version}" if dataset_config.dataset_version != '' else ''
    return dataset_path, dataset_path.split('/')[-1]


def build_model_name(model_type, env_name, dataset_name, env_params, model_params, train_params):
    env_hash = hashlib.md5(str(env_params).encode())
    model_hash = hashlib.md5(str(model_params).encode())
    train_hash = hashlib.md5(str(train_params).encode())
    hash_object = hashlib.md5((env_hash.hexdigest() + model_hash.hexdigest() + train_hash.hexdigest()).encode())
    return f"{model_type}-{env_name}-{dataset_name}-{hash_object.hexdigest()[:8]}"
