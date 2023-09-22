import os
from dataclasses import dataclass, asdict

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

import gym
from gym import Env as GymEnv
from daxbench.core.envs.basic.cloth_env import ClothEnv
from daxbench.core.engine.cloth_simulator import ClothState

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
    render_fps: int = 1
    calc_fps: int = 100 * render_fps  # REVIEW:render_fpsに対して切りが良い値にする
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

    env: str = "daxbench/unfoldClothGymEnv-v0"
    obs_type: str = "image"
    cam_resolution: int = 64
    batch_size: int = 1


class UnfoldClothGymEnv(ClothEnv, GymEnv):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, conf: UnfoldClothGymEnvConf, batch_size, max_steps, aux_reward=False):
        ClothEnv.__init__(self, conf, batch_size, max_steps, aux_reward)
        GymEnv.__init__(self)

        self.metadata["render_fps"] = conf.render_fps

        self.conf = conf

        # observation_spaceの設定:RGB WxHxC
        shape = (conf.cam_resolution, conf.cam_resolution, 3)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            dtype=np.uint8,
            shape=shape,
        )
        self.state: ClothState = None

    def reset(self, key):
        obs, state = super().reset(key)
        self.state: ClothState = state
        obs = self.get_obs()
        return obs, state

    def step(self, actions):
        actions = self.get_pnp_actions(actions)
        for _ in range(self.conf.calc_fps):
            _, reward, done, info = super().step_diff(actions, self.state)
            self.state = info["state"]

        obs = self.get_obs()
        return obs, reward, done, info

    def get_obs(self):
        # RBG画像を返す
        # TODO: バッチ処理にする
        imgs = np.zeros((self.batch_size, self.conf.cam_resolution, self.conf.cam_resolution, 3))
        for idx in range(self.batch_size):
            rgb, depth = self.render(self.state, visualize=False)
            imgs[idx] = rgb

        return imgs

    # エラー履かれたので、実装。意味は不明
    def create_cloth_mask(self, conf):
        N, size = conf.N, conf.size
        cloth_mask = jnp.zeros((N, N))
        cloth_mask = cloth_mask.at[size * 2 : size * 3, size * 2 : size * 4].set(1)

        return cloth_mask


# 単体テスト
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    daxbench_args = UnfoldClothGymEnvConf()
    daxbench_args.batch_size = 3

    env = UnfoldClothGymEnv(daxbench_args, daxbench_args.batch_size, 15)

    for idx in range(5):
        actions = [env.action_space.sample() for _ in range(daxbench_args.batch_size)]
        actions = np.array(actions)
        obs, reward, done, info = env.step(actions)
        print("obs.shape:", obs.shape)
        print("reward:", reward)
        print("done:", done)

        # 観測値を画像で出力
        plt.imsave(f"figures/tmp{idx}.png", obs[0])
