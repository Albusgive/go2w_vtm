import isaaclab
from isaaclab.app import AppLauncher
import gymnasium as gym
from go2w_vtm.locomotion.go2w import *

def test_imports():
    print("检查环境注册...")
    print("\n已注册的环境：")
    for env_id in gym.registry.keys():
        if "Go2W" in env_id:
            print(f"- {env_id}")

if __name__ == "__main__":
    test_imports()