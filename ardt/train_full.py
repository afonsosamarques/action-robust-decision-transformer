import numpy as np

from dataclasses import dataclass
from datasets import load_dataset, load_from_disk
from huggingface_hub import login, list_models
from transformers import DecisionTransformerConfig, Trainer, TrainingArguments

from model.ardt_naive import SingleAgentRobustDT
from model.ardt_full import TwoAgentRobustDT
from model.ardt_utils import DecisionTransformerGymDataCollator

from access_tokens import WRITE_TOKEN


RETURNS_SCALE = 1000.0
CONTEXT_SIZE = 20
N_EPOCHS = 300
WARMUP_EPOCHS = 50


if __name__ == "__main__":
    login(token=WRITE_TOKEN)

    # env and model to run
    envs = {
        0: "walker2d-expert-v2",
        1: "halfcheetah-expert-v2",
    }
    chosen_env = envs[1]

    agent = {
        0: SingleAgentRobustDT,
        1: TwoAgentRobustDT
    }
    chosen_agent = agent[1]
    model_name_prefix = "ardt-" if chosen_agent == TwoAgentRobustDT else "ardt-naive-"

    # dataset to use
    dataset = load_from_disk("./datasets/default_dataset")

    for l1 in [0.05, 0.1, 0.5]:
        for l2 in [3, 8, 15]:
            print(f"Training TwoAgentRobustDT with l1={l1} and l2={l2}")

            # build model
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
                                            max_return=1000) # FIXME completely random, potentially not needed
            model = chosen_agent(config)

            # model naming
            my_env_name = model_name_prefix + chosen_env.split("-")[0]
            models = sorted([m.modelId.split("/")[-1] for m in list_models(author="afonsosamarques")])
            models = [m for m in models if my_env_name in m]
            if len(models) > 0:
                latest_version = [m.split("-")[-1][1:] for m in models][-1]
                new_version = "v" + str(int(latest_version) + 1)
            else:
                new_version = "v0"
            model_name = my_env_name + "-" + new_version + f"-lambda1{l1}" + f"-lambda2{l2}"

            # we use the same hyperparameters as in the authors original implementation, but train for fewer iterations
            training_args = TrainingArguments(
                output_dir="./agents/" + model_name,
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
                hub_model_id=model_name,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=collator,
            )

            trainer.train()
            trainer.save_model()
