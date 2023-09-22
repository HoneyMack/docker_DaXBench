import os
from dataclasses import dataclass, asdict

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

import gym
from gym import Env as GymEnv
from daxbench.core.envs.basic.cloth_env import ClothEnv

my_path = os.path.dirname(os.path.abspath(__file__))


@dataclass
class UnfoldClothGymEnvConf:
    """
    Default configuration class for the 'unfold_cloth_gymenv' environment.
    """

    N: int = 80
    cell_size: float = 1.0 / N
    gravity: float = 0.5
    stiffness: int = 900
    damping: int = 2
    render_fps: int = 30
    calc_fps: int = 20 * render_fps  # REVIEW:render_fpsに対して切りが良い値にする
    dt: float = 1.0 / calc_fps
    max_v: float = 2.0
    small_num: float = 1e-8
    mu: int = 3  # friction
    seed: int = 1
    size: int = int(N / 5.0)
    mem_saving_level: int = 1
    # 0:fast but requires more memory, not recommended
    # 1:lesser memory, but faster
    # 2:much lesser memory but much slower
    task: str = "unfold_cloth_gymenv"
    goal_path: str = f"{my_path}/goals/{task}/goal.npy"
    use_substep_obs: bool = False
    env: str = "unfold_cloth_gymenv"


class UnfoldClothGymEnv(ClothEnv, GymEnv):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, conf: UnfoldClothGymEnvConf, batch_size, max_steps, aux_reward=False):
        ClothEnv.__init__(conf, batch_size, max_steps, aux_reward)
        GymEnv.__init__(self)

        self.metadata["render_fps"] = conf.render_fps
