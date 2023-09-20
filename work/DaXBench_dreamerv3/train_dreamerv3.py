import warnings
import dreamerv3
from dreamerv3 import embodied
from dreamerv3.embodied.envs import from_gym

import gym


# daxbenchの環境まわり
from daxbench.core.envs import UnfoldCloth1Env
from unfold_cloth_gymenv import UnfoldClothGymEnvConf
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
            "logdir": "~/work/DaXBench/logdir/run1",
            "run.train_ratio": 64,
            "run.log_every": 30,  # Seconds
            "batch_size": 16,
            "batch_length": 64,  # 8,16　ならfloat16で動く,float32, 16,64 (xlargeだとfloat16は8,16で動かず8,8のほうがいいかもしれない)
            # "data_loaders":1, #8
            "jax.prealloc": False,
            "jax.logical_cpus": 0,
            "jax.policy_devices": [1],
            "jax.train_devices": [0, 2],
            "encoder.mlp_keys": "$^",
            "decoder.mlp_keys": "$^",
            "encoder.cnn_keys": "image",
            "decoder.cnn_keys": "image",
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
    env = gym.make(daxbench_args.env, args=dedo_args)  # Replace this with your Gym #env.crafter.Env()
    env = from_gym.FromGym(env, obs_key="image")  # Or obs_key='vector'.
    env = dreamerv3.wrap_env(env, config)
    env = embodied.BatchEnv([env], parallel=False)

    # pdb.set_trace()
    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
    replay = embodied.replay.Uniform(config.batch_length, config.replay_size, logdir / "replay")
    args = embodied.Config(**config.run, logdir=config.logdir, batch_steps=config.batch_size * config.batch_length)
    embodied.run.train(agent, env, replay, logger, args)
    # embodied.run.eval_only(agent, env, logger, args)


if __name__ == "__main__":
    main()
