import gym
from multiworld.envs.pygame.point2d import Point2DEnv, Point2DWallEnv
from multiworld.core.flat_goal_env import FlatGoalEnv


def create_pygame_2d():
    env = Point2DEnv(
        images_are_rgb=True,
        target_radius=1.,
        ball_radius=1.,
        render_onscreen=False,
        show_goal=True,
        get_image_base_render_size=(48, 48),
    )
    return FlatGoalEnv(
        env,
        append_goal_to_obs=True,
    )


def create_pygame_box_2d():
    env = Point2DWallEnv(
        action_scale=.1,
        wall_shape='box',
        wall_thickness=2.0,
        images_are_rgb=True,
        render_onscreen=False,
        show_goal=True,
        get_image_base_render_size=(48, 48),
        wall_color='black',
        bg_color='white',
    )
    return FlatGoalEnv(
        env,
        append_goal_to_obs=True,
    )
