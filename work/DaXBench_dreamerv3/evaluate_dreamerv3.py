# dreamerで学習した結果を可視化するプログラム
import pathlib, json
from argparse import ArgumentParser
import jax
import dreamerv3
from dreamerv3 import embodied
from dreamerv3.embodied.envs import from_gym
from daxbench_gymenv.unfold_cloth_gymenv import (
    UnfoldClothGymEnvConf,
    UnfoldClothGymEnv,
)

from classopt import (
    classopt,
    config,
)  # 詳細は以下参照：https://zenn.dev/moisutsu/articles/argument-parser-by-class

import numpy as np
from PIL import Image

import pdb


@classopt(default_long=True)
class Args:
    logdir: str = config(
        default="~/work/DaXBench_dreamerv3/logdir/run24_rgbd",
        type=str,
    )
    eval_size: int = config(default=10, type=int)


def save_images(array, filename):
    batch, iter, w, h, c = array.shape

    # 画像を横に並べるための行列を作成
    image = np.transpose(array, (0, 2, 1, 3, 4))
    image = np.reshape(image, (batch * w, iter * h, c))
    # 画像を保存
    Image.fromarray(np.uint8(image)).save(filename)


def main(args: Args):
    # pathの設定
    logdir = pathlib.Path(args.logdir).expanduser()

    # agentのconfigの設定
    config_dreamerv3 = embodied.Config.load(
        logdir / "config_dreamerv3.json"
    )

    # 環境のconfigの設定
    with open(logdir / "config_daxbench.json", "r") as f:
        conf = json.load(f)
        config_daxbench = UnfoldClothGymEnvConf(**conf)
        config_daxbench.cam_pose = np.array(
            config_daxbench.cam_pose
        )  # listになってしまっているのでnumpyに変換

    # 環境の作成
    env_original = UnfoldClothGymEnv(
        config_daxbench,
        config_daxbench.batch_size,
        max_steps=5,
        aux_reward=True,
    )
    env = from_gym.FromGym(
        env_original, obs_key="image"
    )  # Or obs_key='vector'.
    env = dreamerv3.wrap_env(env, config_dreamerv3)
    env = embodied.BatchEnv([env], parallel=False)

    # agentの作成
    step = embodied.Counter()
    agent = dreamerv3.Agent(
        env.obs_space, env.act_space, step, config_dreamerv3
    )

    eval_rgbs = np.zeros(
        (
            args.eval_size,
            config_daxbench.max_steps,
            config_daxbench.screen_size[0],
            config_daxbench.screen_size[1],
            3,
        )
    )
    # 動作
    with jax.transfer_guard("allow"):
        # 最初の1回目は何もせずにリセットする
        action_reset = {
            "action": env_original.action_space.sample().reshape(
                (1, -1)
            ),
            "reset": np.array([True]),
        }
        obs = env.step(action_reset)

        # 2回目以降は学習したpolicyに従って動作する
        for eval_idx in range(args.eval_size):
            for idx in range(config_daxbench.max_steps):
                action, _ = agent.policy(obs, mode="eval")
                action["reset"] = np.array([False])
                obs = env.step(action)
                
                eval_rgbs[eval_idx, idx] = obs["rgb"][0]

    # 画像を保存
    save_images(eval_rgbs, logdir / "eval_rgbs.png")


if __name__ == "__main__":
    args = Args()
    main(args)
