import gym
from multiworld.core.flat_goal_env import FlatGoalEnv


def create_flat_sawyer_push():
    env = gym.make('SawyerPush-v0')
    env.hide_goal = False
    return FlatGoalEnv(
        env,
        append_goal_to_obs=True,
    )
