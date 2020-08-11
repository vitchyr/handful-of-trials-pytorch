import gym
from multiworld.core.flat_goal_env import FlatGoalEnv


def create_sawyer_push_flat():
    env = gym.make('SawyerPush-v0')
    env.hide_goal = False
    return FlatGoalEnv(
        env,
        append_goal_to_obs=True,
    )
