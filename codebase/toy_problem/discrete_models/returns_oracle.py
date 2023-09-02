from ..toyenv_one import OneStepEnvVOne
from ..toyenv_two import OneStepEnvVTwo
from ..toyenv_three import OneStepEnvVThree


class ReturnsOracle:
    def __init__(self, env_version):
        self.env_version = env_version

    def __call__(self, action):
        if self.env_version == "v1":
            return OneStepEnvVOne.get_wc_returns_for_pr_action(action)
        elif self.env_version == "v2":
            return OneStepEnvVTwo.get_wc_returns_for_pr_action(action)
        elif self.env_version == "v3":
            return OneStepEnvVThree.get_wc_returns_for_pr_action(action)
        else:
            raise ValueError("Invalid environment version: {}".format(self.env_version))
