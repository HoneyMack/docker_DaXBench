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
    #    config_dreamerv3 = embodied.Config(embodied.configs["defaults"])
    #    config_dreamerv3 = config_dreamerv3.update(embodied.configs["medium"])
    #    config_dreamerv3 = config_dreamerv3.update({
    #          "logdir": logdir,
    #    })

    #    config_dreamerv3 = embodied.Flags(config_dreamerv3).parse()

    # 環境のconfigの設定
    #    config_daxbench = UnfoldClothGymEnvConf()
    #    config_daxbench.seed = config_dreamerv3.seed # seedを揃える
    #    config_daxbench.batch_size = 1 # 環境内部でのbatch_sizeは1
    with open(logdir/"config_daxbench.json", "r") as f:
        conf = json.load(f)
        config_daxbench = UnfoldClothGymEnvConf(**conf)
        config_daxbench.cam_pose = np.array(config_daxbench.cam_pose) # listになってしまっているのでnumpyに変換
        #config_daxbench: UnfoldClothGymEnvConf = UnfoldClothGymEnvConf.from_json(f.read())
   
    # 環境の作成
    env = UnfoldClothGymEnv(config_daxbench, config_daxbench.batch_size, max_steps=5,aux_reward=True)
    env = from_gym.FromGym(env, obs_key="image")  # Or obs_key='vector'.
    env = dreamerv3.wrap_env(env, config_dreamerv3)
    env = embodied.BatchEnv([env], parallel=False)

    # agentの作成
    step = embodied.Counter()
    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config_dreamerv3)

    #動作
    eval_rgbs = np.zeros((args.eval_size, config_daxbench.max_steps, config_daxbench.screen_size[0], config_daxbench.screen_size[1], 3))
    obs = env.render()
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