from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from collections import OrderedDict, deque
from time import localtime, strftime
import numpy as np

from dotmap import DotMap
from scipy.io import savemat
from tqdm import trange

import eval_util
from Agent import Agent
from DotmapUtils import get_required_argument
from easy_logger import logger, timer


class MBExperiment:
    def __init__(self, params):
        """Initializes class instance.

        Argument:
            params (DotMap): A DotMap containing the following:
                .sim_cfg:
                    .env (gym.env): Environment for this experiment
                    .task_hor (int): Task horizon
                    .stochastic (bool): (optional) If True, agent adds noise to its actions.
                        Must provide noise_std (see below). Defaults to False.
                    .noise_std (float): for stochastic agents, noise of the form N(0, noise_std^2I)
                        will be added.

                .exp_cfg:
                    .ntrain_iters (int): Number of training iterations to be performed.
                    .nrollouts_per_iter (int): (optional) Number of rollouts done between training
                        iterations. Defaults to 1.
                    .ninit_rollouts (int): (optional) Number of initial rollouts. Defaults to 1.
                    .policy (controller): Policy that will be trained.

                .log_cfg:
                    .logdir (str): Parent of directory path where experiment data will be saved.
                        Experiment will be saved in logdir/<date+time of experiment start>
                    .nrecord (int): (optional) Number of rollouts to record for every iteration.
                        Defaults to 0.
                    .neval (int): (optional) Number of rollouts for performance evaluation.
                        Defaults to 1.
        """

        # Assert True arguments that we currently do not support
        assert params.sim_cfg.get("stochastic", False) == False

        self.env = get_required_argument(params.sim_cfg, "env", "Must provide environment.")
        self.task_hor = get_required_argument(params.sim_cfg, "task_hor", "Must provide task horizon.")
        self.agent = Agent(DotMap(env=self.env, noisy_actions=False))

        self.ntrain_iters = get_required_argument(
            params.exp_cfg, "ntrain_iters", "Must provide number of training iterations."
        )
        self.nrollouts_per_iter = params.exp_cfg.get("nrollouts_per_iter", 1)
        self.ninit_rollouts = params.exp_cfg.get("ninit_rollouts", 1)
        self.max_num_saved_trajs = params.exp_cfg.get("max_num_saved_trajs", 100
        )
        self.snapshot_period = params.exp_cfg.get("snapshot_period", 20)
        self.policy = get_required_argument(params.exp_cfg, "policy", "Must provide a policy.")

        directory = get_required_argument(params.log_cfg, "logdir", "Must provide log parent directory.")
        exp_name = '{}-{}'.format(
            os.path.dirname(directory).split('/')[-1],
            strftime("%Y-%m-%d--%H-%M-%S", localtime()),
        )
        self.logdir = os.path.join(directory, exp_name)
        self.nrecord = params.log_cfg.get("nrecord", 0)
        self.neval = params.log_cfg.get("neval", 1)

    def run_experiment(self):
        """Perform experiment.
        """
        os.makedirs(self.logdir, exist_ok=True)

        # traj_obs, traj_acs, traj_rets, traj_rews = [], [], [], []
        traj_obs = deque(maxlen=self.max_num_saved_trajs)
        traj_acs = deque(maxlen=self.max_num_saved_trajs)
        traj_rets = deque(maxlen=self.max_num_saved_trajs)
        traj_rews = deque(maxlen=self.max_num_saved_trajs)

        # Perform initial rollouts
        samples = []
        for i in range(self.ninit_rollouts):
            sample, infos = self.agent.sample(
                self.task_hor, self.policy
            )
            samples.append(sample)
            traj_obs.append(samples[-1]["obs"])
            traj_acs.append(samples[-1]["ac"])
            traj_rews.append(samples[-1]["rewards"])

        if self.ninit_rollouts > 0:
            self.policy.train(
                [sample["obs"] for sample in samples],
                [sample["ac"] for sample in samples],
                [sample["rewards"] for sample in samples]
            )

        # Training loop
        timer.return_global_times = True
        for i in trange(self.ntrain_iters):
            timer.reset()
            print("####################################################################")
            print("Starting training iteration %d." % (i + 1))

            iter_dir = os.path.join(self.logdir, "train_iter%d" % (i + 1))
            os.makedirs(iter_dir, exist_ok=True)

            samples = []
            eval_infos = []

            timer.start_timer('exploration')
            for rollout_i in range(max(self.neval, self.nrollouts_per_iter)):
                if i % self.snapshot_period == 0:
                    video_path = os.path.join(
                        iter_dir,
                        'rollout{}.mp4'.format(rollout_i)
                    )
                    save_path = os.path.join(
                        iter_dir,
                        'model.pt',
                    )
                    self.policy.save_snapshot(save_path)
                else:
                    video_path = None
                sample, eval_info = self.agent.sample(
                    self.task_hor, self.policy, record_fname=video_path,
                )
                samples.append(sample)
                eval_infos.append(eval_info)
            timer.stop_timer('exploration')
            traj_obs.extend([sample["obs"] for sample in samples[:self.nrollouts_per_iter]])
            traj_acs.extend([sample["ac"] for sample in samples[:self.nrollouts_per_iter]])
            traj_rets.extend([sample["reward_sum"] for sample in samples[:self.neval]])
            traj_rews.extend([sample["rewards"] for sample in samples[:self.nrollouts_per_iter]])
            samples = samples[:self.nrollouts_per_iter]

            self.policy.dump_logs(self.logdir, iter_dir)
            savemat(
                os.path.join(self.logdir, "logs.mat"),
                {
                    "observations": traj_obs,
                    "actions": traj_acs,
                    "returns": traj_rets,
                    "rewards": traj_rews
                }
            )
            # Delete iteration directory if not used
            if len(os.listdir(iter_dir)) == 0:
                os.rmdir(iter_dir)

            timer.start_timer('model_training')
            if i < self.ntrain_iters - 1:
                train_logs = self.policy.train(
                    [sample["obs"] for sample in samples],
                    [sample["ac"] for sample in samples],
                    [sample["rewards"] for sample in samples]
                )
            else:
                train_logs = {}
            timer.stop_timer('model_training')

            logger.record_tabular('epoch', i)
            logger.record_tabular(
                'exploration/num steps total',
                i * self.nrollouts_per_iter * self.task_hor,
            )
            eval_logs = generate_logging_info(samples, eval_infos)
            logger.record_dict(train_logs, prefix='train')
            logger.record_dict(eval_logs, prefix='evaluation/eval/')
            logger.record_dict(_get_epoch_timings())
            logger.dump_tabular()


def generate_logging_info(samples, eval_infos):
    stats = OrderedDict()
    stats['Average Returns'] = np.mean([sample["reward_sum"] for sample in samples])
    stats.update(
        eval_util.create_stats_ordered_dict(
            'Rewards',
            np.array([sample["rewards"] for sample in samples])
        )
    )
    if eval_infos and eval_infos[0]:
        eval_keys = eval_infos[0][0].keys()
        for key in eval_keys:
            stats.update(eval_util.extract_stats(
                eval_infos,
                key,
            ))
    return stats


def _get_epoch_timings():
    times_itrs = timer.get_times()
    times = OrderedDict()
    epoch_time = 0
    for key in sorted(times_itrs):
        time = times_itrs[key]
        epoch_time += time
        times['time/{} (s)'.format(key)] = time
    return times
