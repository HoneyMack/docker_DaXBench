# dreamerで学習した結果を可視化するプログラム
import pathlib,json
from argparse import ArgumentParser
import jax
import dreamerv3
from dreamerv3 import embodied
from dreamerv3.embodied.envs import from_gym 
from daxbench_gymenv.unfold_cloth_gymenv import UnfoldClothGymEnvConf, UnfoldClothGymEnv


from classopt import classopt, config # 詳細は以下参照：https://zenn.dev/moisutsu/articles/argument-parser-by-class

import numpy as np
from PIL import Image

@classopt(default_long=True)
class Args:
    logdir: str = config(default="~/work/DaXBench_dreamerv3/logdir/run20_rgbd",type=str)
    eval_size: int = config(default=10,type=int)



def save_images(array, filename):
    batch, iter, w, h, c = array.shape
    # # 画像を横に並べるための行列を作成
    # rows = []
    # for i in range(iter):
    #     row = []
    #     for j in range(batch):
    #         row.append(array[j, i])
    #     rows.append(np.concatenate(row, axis=1))
    # # 行列を縦に並べる
    # image = np.concatenate(rows, axis=0)
    
    # 上記の処理の高速化ver
    # 画像を横に並べるための行列を作成
    image = np.transpose(array, (0, 2, 1, 3, 4))
    image = np.reshape(image, (batch * w, iter * h, c))
    # 画像を保存
    Image.fromarray(np.uint8(image)).save(filename)

def main(args: Args):
    # pathの設定
    logdir = pathlib.Path(args.logdir).expanduser()
    
    # agentのconfigの設定
    config_dreamerv3 = embodied.Config.load(logdir/"config_dreamerv3.json")

    # 環境のconfigの設定
    with open(logdir/"config_daxbench.json", "r") as f:
        conf = json.load(f)
        config_daxbench = UnfoldClothGymEnvConf(**conf)
        config_daxbench.cam_pose = np.array(config_daxbench.cam_pose) # listになってしまっているのでnumpyに変換
   
    # 環境の作成
    env_original = UnfoldClothGymEnv(config_daxbench, config_daxbench.batch_size, max_steps=5,aux_reward=True)
    env = from_gym.FromGym(env_original, obs_key="image")  # Or obs_key='vector'.
    # with jax.transfer_guard("allow"):
    #     obs = env.render()
    env = dreamerv3.wrap_env(env, config_dreamerv3)
    env = embodied.BatchEnv([env], parallel=False)

    action = {"action": env_original.action_space.sample(),"reset":True}
    obs = env.step(action)
    # agentの作成
    step = embodied.Counter()
    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config_dreamerv3)


    #動作
    eval_rgbs = np.zeros((args.eval_size, config_daxbench.max_steps, config_daxbench.screen_size[0], config_daxbench.screen_size[1], 3))
    with jax.transfer_guard("allow"):
        for eval_idx in range(args.eval_size):    
            for idx in range(5):
                action = agent.policy(obs,mode="eval")
                obs, reward, done, info = env.step(action)
                eval_rgbs[eval_idx, idx] = obs
        
    # 画像を保存
    save_images(eval_rgbs, logdir/"eval_rgbs.png")
    
if __name__ == "__main__":
    args = Args()
    main(args)