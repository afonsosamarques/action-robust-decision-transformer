# Action Robust Decision Transformer

To install required packages, simply run
```bash
pip install -r action-robust-decision-transformer/requirements.txt
```


### Instructions

Our code sits under the main directory `codebase` and can be broken down as follows. All our proposed models and related training, as well as related utilities, is placed under the directory `ardt`. For evaluation, all the necessary code can be found under `evaluation_protocol`. Directory `baselines` contains modified code for the baseline data collection policies, and `toy_problem` contains all models, training and evaluation for three discrete, one-step toy environments.

Run code as modules and not as functions. For example, to run a test run of the pipeline for ardt using some example config, from the `codebase` directory:
```bash
python3 -m ardt.pipeline --config_name pipeline-example
```

Concretely, if you want to launch a training run set your configurations in a yaml file following the provided examples, and place it the file under `ardt/run-configs`. Make sure the required dataset is placed under `ardt/datasets`.

To launch an evaluation run, you will also require a yaml configuration file. Place it under `evaluation_protocol/run-configs` and run, for example:
```bash
python3 -m evaluation_protocol.evaluate --config_name evaluation-batch-envadv-example
```

Finally, the toy problem can also be run end-to-end. Again a yaml configuration file is necessary, but both training and evaluation can be run using the default evaluation protocol by issuing the command:
```bash
python3 -m toy_problem.pipeline --config_name ardt-multipart-toy-example
```


### References

For building our models and training pipelines:
* Wolf, Thomas, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac et al. "Transformers: State-of-the-art natural language processing." In Proceedings of the 2020 conference on empirical methods in natural language processing: system demonstrations, pp. 38-45. 2020. Available at: https://huggingface.co/

For logging and monitoring training:
* Weights and Biases (2020). Weights & Biases. Weights and Biases Inc. Available at: https://www.wandb.com/

For our online data collection policies:
* Tessler, Chen, Yonathan Efroni, and Shie Mannor. "Action robust reinforcement learning and applications in continuous control." In International Conference on Machine Learning, pp. 6215-6224. PMLR, 2019.

* Kamalaruban, Parameswaran, Yu-Ting Huang, Ya-Ping Hsieh, Paul Rolland, Cheng Shi, and Volkan Cevher. "Robust reinforcement learning via adversarial training with langevin dynamics." Advances in Neural Information Processing Systems 33 (2020): 8127-8138.

Other datasets:
* Fu, Justin, Aviral Kumar, Ofir Nachum, George Tucker, and Sergey Levine. "D4rl: Datasets for deep data-driven reinforcement learning." arXiv preprint arXiv:2004.07219 (2020).
