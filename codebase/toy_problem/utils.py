import datetime
import subprocess
import os
import random

import numpy as np
import torch

from dataclasses import dataclass
from pydantic import BaseModel, Field

from .discrete_models.dt_vanilla import VanillaDT
from .discrete_models.ardt_vanilla import VanillaARDT
from .discrete_models.ardt_multipart import MultipartARDT
from transformers import DecisionTransformerConfig


def find_root_dir():
    try:
        root_dir = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode('utf-8')
    except Exception as e:
        root_dir = os.getcwd()[:os.getcwd().find('action-robust-decision-transformer')+len('action-robust-decision-transformer')]
    return root_dir + ('' if root_dir.endswith('action-robust-decision-transformer') else '/action-robust-decision-transformer') + "/codebase/toy_problem"


def load_agent(agent_type):
    if agent_type == "dt":
        return VanillaDT, False
    elif agent_type == "ardt-vanilla" or agent_type == "ardt_vanilla":
        return VanillaARDT, True
    elif agent_type == "ardt-multipart" or agent_type == "ardt_multipart":
        return MultipartARDT, False
    else:
        raise Exception(f"Agent type {agent_type} not available.")
    

def load_model(model_type, model_to_use, model_path):
    if model_type == "dt":
        config = DecisionTransformerConfig.from_pretrained(model_path)
        model = VanillaDT(config)
        return model.from_pretrained(model_path), False
    elif model_type == "ardt-vanilla" or model_type == "ardt_vanilla":
        config = DecisionTransformerConfig.from_pretrained(model_path)
        model = VanillaARDT(config)
        return model.from_pretrained(model_path), True
    elif model_type == "ardt-multipart" or model_type == "ardt_multipart":
        config = DecisionTransformerConfig.from_pretrained(model_path)
        model = MultipartARDT(config)
        return model.from_pretrained(model_path), True
    else:
        raise Exception(f"Model {model_to_use} of type {model_type} not available.")
    

def set_seed_everywhere(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if env is not None:
        env.seed = seed
        env.action_space.seed = seed


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
    

def build_model_name(model_type, dataset_name):
    datetime_encoding = datetime.datetime.now().strftime("%d%m_%H%M")
    return f"{model_type}-{dataset_name}-{datetime_encoding}"


##########################################################################
class EnvironmentConfig(BaseModel):
    env_type: str = Field(...)
    max_ep_len: int = Field(...)
    returns_scale: int = Field(...)
    max_ep_return: int = Field(...)


class DatasetConfig(BaseModel):
    is_local: list[bool] = Field(...)
    online_policy_names : list[str] = Field(...)
    dataset_types: list[str] = Field(...)
    dataset_versions: list[str] = Field(...)


class ModelConfig(BaseModel):
    agent_type: str = Field(...)
    context_size: list[int] = Field(...)


class TrainingConfig(BaseModel):
    train_steps: int = Field(...)
    train_batch_size: int = Field(...)
    learning_rate: list[float] = Field(...)
    weight_decay: list[float] = Field(...)
    max_grad_norm: list[float] = Field(...)
    warmup_steps: list[int] = Field(...)


class EvaluationConfig(BaseModel):
    is_eval: bool = Field(...)
    eval_type: str = Field(...)
    eval_target_return: int = Field(...)
    eval_iters: int = Field(...)
    adv_model_names: list[str] = Field(...)
    adv_model_types: list[str] = Field(...)


class AdminConfig(BaseModel):
    wandb_project: str = Field(...)
    hf_project: str = Field(...)
    run_type: str = Field(...)
    is_verbose: bool = Field(...)
    print_tracebacks: bool = Field(...)


@dataclass
class PipelineConfig:
    environment_config: EnvironmentConfig = None
    dataset_config: DatasetConfig = None
    model_config: ModelConfig = None
    training_config: TrainingConfig = None
    evaluation_config: EvaluationConfig = None
    admin_config: AdminConfig = None


def check_pipelinerun_config(config, do_checks=True):
    # field existence/type validation
    environment_config = EnvironmentConfig(**(config['environment_config']))
    dataset_config = DatasetConfig(**(config['dataset_config']))
    model_config = ModelConfig(**(config['model_config']))
    training_config = TrainingConfig(**(config['training_config']))
    evaluation_config = EvaluationConfig(**(config['evaluation_config']))
    admin_config = AdminConfig(**(config['admin_config']))
    pipeline_config = PipelineConfig(**{
        'environment_config': environment_config,
        'dataset_config': dataset_config,
        'model_config': model_config,
        'training_config': training_config,
        'evaluation_config': evaluation_config,
        'admin_config': admin_config
    })
    return pipeline_config
