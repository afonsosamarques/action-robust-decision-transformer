import datetime

from dataclasses import dataclass
from pydantic import BaseModel, Field

from ..model.ardt_simplest import SimpleRobustDT
from ..model.ardt_vanilla import SingleAgentRobustDT
from ..model.ardt_full import TwoAgentRobustDT
from ..model.trainable_dt import TrainableDT

from .helpers import find_root_dir


##########################################################################
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
    

def load_agent(agent_type):
    if agent_type == "dt":
        return TrainableDT
    elif agent_type == "ardt-simplest" or agent_type == "ardt_simplest":
        return SimpleRobustDT
    elif agent_type == "ardt-vanilla" or agent_type == "ardt_vanilla":
        return SingleAgentRobustDT
    elif agent_type == "ardt-full" or agent_type == "ardt_full":
        return TwoAgentRobustDT
    else:
        raise Exception(f"Agent type {agent_type} not available.")


def build_dataset_path(dataset_config, env_name, is_local=True, hf_project="afonsosamarques"):
    dataset_dir = find_root_dir() + "/datasets" if is_local else f"{hf_project}"
    dataset_path = f"{dataset_dir}/{dataset_config.online_policy_name}_{dataset_config.dataset_type}_{env_name}"
    dataset_path += f"_{dataset_config.dataset_version}" if dataset_config.dataset_version != '' else ''
    return dataset_path, dataset_path.split('/')[-1]


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
    is_local: bool = Field(...)
    online_policy_name : str = Field(...)
    dataset_type: str = Field(...)
    dataset_version: str = Field(...)


class ModelConfig(BaseModel):
    agent_type: str = Field(...)
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


def check_pipelinerun_config(config):
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
    assert pipeline_config.environment_config.env_type in ['halfcheetah', 'hopper', 'walker2d'], "Environment name needs to be either 'halfcheetah', 'hopper' or 'walker2d'."
    assert pipeline_config.dataset_config.online_policy_name in ['d4rl', 'rarl', 'arrl_prmdp', 'arrl_nrmdp', 'dataset_combo', 'randagent', 'ppo', 'trpo'], "Online policy name needs to be either 'd4rl', 'rarl', 'arrl_prmdp', 'arrl_nrmdp', 'dataset_combo' or 'randagent'."
    assert pipeline_config.dataset_config.dataset_type in ['train', 'test', 'expert', 'mixed'], "Dataset type needs to be either 'train' or 'test' or 'expert' or 'mixed'."
    assert pipeline_config.model_config.agent_type in ['dt', 'ardt-simplest', 'ardt-vanilla', 'ardt-full']
    assert pipeline_config.admin_config.wandb_project in ['afonsosamarques', 'ARDT-Project', 'ARDT-Internal'], "Wandb project needs to be either 'afonsosamarques' or 'ARDT-Project' or 'ARDT-Internal'."
    assert pipeline_config.admin_config.hf_project in ['afonsosamarques', 'ARDT-Project', 'ARDT-Internal'], "Wandb project needs to be either 'afonsosamarques' or 'ARDT-Project' or 'ARDT-Internal'."
    assert pipeline_config.admin_config.run_type in ['core', 'pipeline', 'test'], "Run type needs to be either 'core', 'pipeline' or 'test'."
    if pipeline_config.evaluation_config.is_eval:
        assert pipeline_config.evaluation_config.eval_type in ['no_adv', 'env_adv', 'agent_adv'], "Evaluation type needs to be either 'no_adv', 'env_adv' or 'agent_adv'."
        if pipeline_config.evaluation_config.eval_type == 'ardt-agent_adv':
            assert len(pipeline_config.evaluation_config.adv_model_names) > 0, "There need to be at least one adversarial model."
            assert len(pipeline_config.evaluation_config.adv_model_names) == len(pipeline_config.evaluation_config.adv_model_types), "There need to be as many adversarial model names as adversarial model types."
    return pipeline_config
