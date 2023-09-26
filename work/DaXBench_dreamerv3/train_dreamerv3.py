import warnings
import dreamerv3
from dreamerv3 import embodied
from dreamerv3.embodied.envs import from_gym

import gym
import jax


# daxbenchの環境まわり
from daxbench.core.envs import UnfoldCloth1Env
from daxbench_gymenv.unfold_cloth_gymenv import UnfoldClothGymEnvConf, UnfoldClothGymEnv
import json
from argparse import ArgumentParser

import pdb


def main():
    warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

    # See configs.yaml for all options.
    config = embodied.Config(dreamerv3.configs["defaults"])
    config = config.update(dreamerv3.configs["medium"])
    config = config.update(
        {
            "logdir": "~/work/DaXBench/logdir/run8_rgbd",
            "run.train_ratio": 64,
            "run.log_every": 120,  # Seconds
            "batch_size": 8,
            "batch_length": 16,  # 8,16　ならfloat16で動く,float32, 16,64 (xlargeだとfloat16は8,16で動かず8,8のほうがいいかもしれない)
            # "data_loaders": 1,  # 8
            "jax.prealloc": False,
            "jax.logical_cpus": 0,
            "jax.policy_devices": [0],
            "jax.train_devices": [1],
            "encoder.mlp_keys": "$^",
            "decoder.mlp_keys": "$^",
            "encoder.cnn_keys": "image|rgb|depth",
            "decoder.cnn_keys": "image|rgb|depth",
            # 'jax.platform': 'cpu',
        }
    )

    # メモ：
    # resolution 128*128,model large, float32, batch_size 16, batch_length 64で動いた
    config = embodied.Flags(config).parse()

    logdir = embodied.Path(config.logdir)
    step = embodied.Counter()
    logger = embodied.Logger(
        step,
        [
            embodied.logger.TerminalOutput(),
            embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
            embodied.logger.TensorBoardOutput(logdir),
            # embodied.logger.WandBOutput(logdir.name, config),
            # embodied.logger.MLFlowOutput(logdir.name),
        ],
    )

    # print(config)

    daxbench_args = UnfoldClothGymEnvConf()
    daxbench_args.seed = config.seed
    # daxbench_args.batch_size = config.batch_size  # 環境(dacbench)で生成するbatch_sizeとdreamerv3側で生成するbatch_sizeを一致させる
    # もしかしたら↑で動かないかも。そのときは↓を使う
    daxbench_args.batch_size = 1  # 各環境内部でのbatch_sizeは1　TODO:環境内でbatch化したほうが明らかに高速になるので、そうなるように改良

    # env = gym.make(daxbench_args.env, args=daxbench_args)  # Replace this with your Gym #env.crafter.Env()
    env = UnfoldClothGymEnv(daxbench_args, daxbench_args.batch_size, 15)
    env = from_gym.FromGym(env, obs_key="image")  # Or obs_key='vector'.
    env = dreamerv3.wrap_env(env, config)
    env = embodied.BatchEnv([env], parallel=False)

    # pdb.set_trace()
    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
    replay = embodied.replay.Uniform(config.batch_length, config.replay_size, logdir / "replay")
    args = embodied.Config(**config.run, logdir=config.logdir, batch_steps=config.batch_size * config.batch_length)

    with jax.transfer_guard("allow"):
        embodied.run.train(agent, env, replay, logger, args)
    # embodied.run.eval_only(agent, env, logger, args)


if __name__ == "__main__":
    main()
