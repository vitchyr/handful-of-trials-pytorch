from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

from dotmap import DotMap

from MBExperiment import MBExperiment
from MPC import MPC
from config import create_config
import env # We run this so that the env is registered

import torch
import numpy as np
import random
import tensorflow as tf
from easy_logger import logger

DIRECTORIES_TO_SAVE = [
    '/home/vitchyr/git/pets-pytorch/',
    '/home/vitchyr/git/multiworld/',
    '/home/vitchyr/git/easy-logger/',
]


def set_global_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    tf.set_random_seed(seed)


def main(env, ctrl_type, ctrl_args, overrides, logdir, seed):
    set_global_seeds(seed)

    ctrl_args = DotMap(**{key: val for (key, val) in ctrl_args})
    cfg = create_config(env, ctrl_type, ctrl_args, overrides, logdir)
    cfg.pprint()

    assert ctrl_type == 'MPC'

    cfg.exp_cfg.exp_cfg.policy = MPC(cfg.ctrl_cfg)
    exp = MBExperiment(cfg.exp_cfg)

    os.makedirs(exp.logdir)
    cfg_dict = cfg.toDict()
    cfg_dict['seed'] = seed
    with open(os.path.join(exp.logdir, "config.txt"), "w") as f:
        f.write(pprint.pformat(cfg.toDict()))

    logger.set_snapshot_dir(exp.logdir)
    print("snapshot dir: ", logger.get_snapshot_dir())
    logger.save_git_snapshot(DIRECTORIES_TO_SAVE)
    logger.save_main_script()
    tabular_log_path = os.path.join(exp.logdir, 'progress.csv')
    logger.add_tabular_output(tabular_log_path)
    variant_log_path = os.path.join(exp.logdir, 'variant.json')
    logger.log_variant(variant_log_path, cfg.toDict())
    logger.push_prefix('[%s] ' % exp.logdir)

    exp.run_experiment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', type=str, required=True,
                        help='Environment name: select from [cartpole, reacher, pusher, halfcheetah]')
    parser.add_argument('-ca', '--ctrl_arg', action='append', nargs=2, default=[],
                        help='Controller arguments, see https://github.com/kchua/handful-of-trials#controller-arguments')
    parser.add_argument('-o', '--override', action='append', nargs=2, default=[],
                        help='Override default parameters, see https://github.com/kchua/handful-of-trials#overrides')
    parser.add_argument('-logdir', type=str, default='log',
                        help='Directory to which results will be logged (default: ./log)')
    parser.add_argument('-seed', type=int, default=None,
                        help='seed. randomly chosen if not set')
    args = parser.parse_args()

    if args.seed is None:
        seed = random.randint(0, 99999999)
    else:
        seed = args.seed
    main(args.env, "MPC", args.ctrl_arg, args.override, args.logdir, seed)
