from gym.envs.registration import register
from .unfold_cloth_gymenv import UnfoldClothGymEnv

# daxbenchのgym環境を登録
register(
    id="daxbench/unfoldClothGymEnv-v0",
    entry_point="gymenv.unfold_cloth_gymenv:UnfoldClothGymEnv",
)
