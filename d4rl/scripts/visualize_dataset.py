import argparse
import d4rl
import gym
import os

# os.environ[
#     'LD_LIBRARY_PATH'] = '$LD_LIBRARY_PATH:/home/wenyongyan/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH:/usr/lib/nvidia'
# os.environ['LD_PRELOAD'] = '$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libGLEW.so'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="maze2d-randMaze0S40-ac-v0")
    args = parser.parse_args()

    env = gym.make(args.env_name)

    # dataset = env.get_dataset()
    # if "infos/qpos" not in dataset:
    #     raise ValueError("Only MuJoCo-based environments can be visualized")
    # qpos = dataset["infos/qpos"]
    # qvel = dataset["infos/qvel"]
    # rewards = dataset["rewards"]
    # actions = dataset["actions"]

    # NSAMPLES = 1000000
    # import matplotlib.pyplot as plt
    # plt.scatter(qpos[:NSAMPLES, 0], qpos[:NSAMPLES, 1])
    # plt.axis('equal')
    # plt.show()

    env.reset()
    # env.set_state(qpos[0], qvel[0])
    # [env.render() for _ in range(100)]
    import time

    for i in range(100):
        env.render()
        time.sleep(1)

    time.sleep(1)
    for t in range(qpos.shape[0]):
        env.set_state(qpos[t], qvel[t])
        env.render()
