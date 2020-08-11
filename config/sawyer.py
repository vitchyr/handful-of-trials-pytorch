from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import torch

from DotmapUtils import get_required_argument
from config.gcac_model import GcacModel

TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class SawyerPushConfigModule:
    ENV_NAME = 'SawyerPushFlat-v0'
    TASK_HORIZON = 100
    NTRAIN_ITERS = 15
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 25
    MODEL_IN, MODEL_OUT = 6, 4
    GP_NINDUCING_POINTS = 200

    def __init__(self):
        self.ENV = gym.make(self.ENV_NAME)
        self.NN_TRAIN_CFG = {"epochs": 5}
        self.OPT_CFG = {
            "Random": {
                "popsize": 2000
            },
            "CEM": {
                "popsize": 400,
                "num_elites": 40,
                "max_iters": 5,
                "alpha": 0.1
            }
        }

    @staticmethod
    def obs_preproc(obs):
        return obs[:, :4]

    @staticmethod
    def obs_postproc(obs, pred):
        new_next_state = pred + obs[:, :4]
        goals = obs[:, 4:]
        new_next_obs = torch.cat((new_next_state, goals), dim=1)
        return new_next_obs

    @staticmethod
    def targ_proc(obs, next_obs):
        return (next_obs - obs)[:, :4]

    @staticmethod
    def goal_proc(obs):
        return obs[:, 4:]

    @staticmethod
    def obs_cost_fn(obs):
        positions = obs[:, :4]
        goals = obs[:, 4:]
        deltas = (positions - goals)
        distances = (deltas ** 2).sum(dim=1).sqrt()
        return (distances >= 0.6).type(obs.dtype)

    @staticmethod
    def ac_cost_fn(acs):
        return 0.01 * (acs ** 2).sum(dim=1)

    def nn_constructor(self, model_init_cfg):

        ensemble_size = get_required_argument(model_init_cfg, "num_nets", "Must provide ensemble size")

        load_model = model_init_cfg.get("load_model", False)

        assert load_model is False, 'Has yet to support loading model'

        model = GcacModel(
            ensemble_size,
            self.MODEL_IN,
            self.MODEL_OUT * 2,  # * 2 b/c we output mean AND variance
            hidden_size=64,
        ).to(TORCH_DEVICE)

        model.optim = torch.optim.Adam(model.parameters(), lr=0.001)

        return model


CONFIG_MODULE = SawyerPushConfigModule
