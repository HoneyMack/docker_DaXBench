import os
from dataclasses import dataclass, asdict
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

import gym
from gym import Env as GymEnv
from daxbench.core.envs.basic.cloth_env import ClothEnv
from daxbench.core.engine.cloth_simulator import ClothState
from daxbench.core.engine.pyrender.py_render import BasicPyRenderer

import pathlib


file_dir = pathlib.Path(__file__).parent.absolute()


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
    calc_fps: int = 1000 * render_fps  # REVIEW:render_fpsに対して切りが良い値にする
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
    goal_path: str = file_dir.joinpath("goals", task, "goal.npy")
    use_substep_obs: bool = False

    env: str = "daxbench/unfoldClothGymEnv-v0"
    obs_type: str = "image"
    batch_size: int = 1
    screen_size: Tuple[int, int] = (128, 128)
    cam_pose: np.ndarray = BasicPyRenderer.look_at(np.array([0.5, 0.5, 1.0]), np.array([0.5, 0.5, 0]))


class UnfoldClothGymEnv(ClothEnv, GymEnv):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, conf: UnfoldClothGymEnvConf, batch_size, max_steps, aux_reward=False):
        ClothEnv.__init__(self, conf, batch_size, max_steps, aux_reward)
        GymEnv.__init__(self)

        self.metadata["render_fps"] = conf.render_fps

        self.conf = conf

        # observation_spaceの設定:RGB WxHxC
        shape = (conf.screen_size[0], conf.screen_size[1], 3)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            dtype=np.uint8,
            shape=shape,
        )
        self.state: ClothState = None

        self.reset = self.build_reset()

    # def reset(self, key):
    #     obs, state = ClothEnv.reset(self, key)
    #     self.state: ClothState = state
    #     obs = self._get_obs()
    #     return obs, state

    def step(self, actions):
        # actions = self.get_pnp_actions(actions, self.state)
        obs, reward, done, info = self.step_diff(actions, self.state)
        # for _ in range(self.conf.calc_fps):
        #     _, reward, done, info = self.simulator.step_jax(self, actions, self.state)
        #     self.state = info["state"]

        obs = self._get_obs()
        return obs, reward, done, info

    def _get_obs(self):
        # RBG画像を返す
        # TODO: バッチ処理にする
        imgs = np.zeros((self.batch_size, self.conf.screen_size[0], self.conf.screen_size[1], 3), dtype=np.uint8)
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

    def random_fold(self, state, key, step=10):
        num_particle = state.x.shape[1]
        batch_idx = jnp.arange(state.x.shape[0])
        for i in range(step):
            st_point = np.random.randint(0, num_particle, size=(state.x.shape[0],))
            ed_point = np.random.randint(0, num_particle, size=(state.x.shape[0],))

            actions = jnp.concatenate((state.x[batch_idx, st_point], state.x[batch_idx, ed_point]), axis=-1)
            _, _, _, info = self.step_diff(actions, state)
            state = info["state"]
        return state

    def build_reset(self):
        init_state = self.simulator.reset_jax()

        def reset(key):
            key, _ = random.split(key)
            new_x = init_state.x + random.normal(key, init_state.x.shape) * 0.0001
            state = init_state._replace(x=new_x)
            state = self.random_fold(state, key, step=2)
            self.state: ClothState = state
            return self._get_obs(), state

        return reset


# 単体テスト
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    daxbench_args = UnfoldClothGymEnvConf()
    daxbench_args.batch_size = 3
    daxbench_args.cam_pose: np.ndarray = BasicPyRenderer.look_at(np.array([0.5, 0.5, 0.8]), np.array([0.501, 0.5, 0]))
    print(daxbench_args.cam_pose)

    env = UnfoldClothGymEnv(daxbench_args, daxbench_args.batch_size, 15)

    # リセット
    key = random.PRNGKey(0)
    obs, state = env.reset(key)

    for idx in range(10):
        actions = [env.action_space.sample() for _ in range(daxbench_args.batch_size)]
        actions = jnp.array(actions)
        obs, reward, done, info = env.step(actions)
        print("obs.shape:", obs.shape)
        print("reward:", reward)
        print("done:", done)

        # 観測値を画像で出力
        save_file = file_dir.joinpath("figures", f"tmp{idx}.png")
        plt.imsave(save_file, obs[0])
