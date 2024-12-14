import gym
import logging
from d4rl.pointmaze import waypoint_controller
from d4rl.pointmaze import maze_model, maze_layouts
import numpy as np
import pickle
import gzip
import h5py
import os
import argparse


def reset_data():
    return {
        "states": [],
        "actions": [],
        "terminals": [],
        "rewards": [],
        "infos/goal": [],
        "infos/qpos": [],
        "infos/qvel": [],
    }


def append_data(data, s, a, tgt, done, env_data):
    data["states"].append(s)
    data["actions"].append(a)
    data["rewards"].append(0.0)
    data["terminals"].append(done)
    data["infos/goal"].append(tgt)
    data["infos/qpos"].append(env_data.qpos.ravel().copy())
    data["infos/qvel"].append(env_data.qvel.ravel().copy())


def npify(data):
    for k in data:
        if k == "terminals":
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Render trajectories")
    parser.add_argument("--noisy", action="store_true", help="Noisy actions")
    parser.add_argument(
        "--env_name", type=str, default="maze2d-randMaze0S20-ac-v0", help="Maze type"
    )
    parser.add_argument(
        "--num_samples", type=int, default=int(150000), help="Num samples to collect"
    )
    parser.add_argument(
        "--data_dir", type=str, default=".", help="Base directory for dataset"
    )
    parser.add_argument(
        "--batch_idx",
        type=int,
        default=int(0),
        help="(Optional) Index of generated data batch",
    )
    args = parser.parse_args()

    env = gym.make(args.env_name)
    maze = env.str_maze_spec
    max_episode_steps = env._max_episode_steps

    controller = waypoint_controller.WaypointController(maze)
    env = maze_model.MazeEnv(maze)

    env.set_target()
    s = env.reset()
    act = env.action_space.sample()
    done = False

    data = reset_data()
    ts, cnt, sum = 0, 0, 0
    for _ in range(args.num_samples):
        position = s[0:2]
        velocity = s[2:4]
        act, done = controller.get_action(position, velocity, env._target)
        if args.noisy:
            act = act + np.random.randn(*act.shape) * 0.5

        act = np.clip(act, -1.0, 1.0)
        if ts >= max_episode_steps:
            done = True
        append_data(data, s, act, env._target, done, env.sim.data)

        ns, _, _, _ = env.step(act)

        ts += 1

        if done:
            save_data(args, data, cnt)
            sum += ts
            print(sum)
            cnt += 1
            data = reset_data()
            env.set_target()
            s = env.reset()
            ts = 0
        else:
            s = ns

        if args.render:
            env.render()


def save_data(args, data, idx):
    # save_video("seq_{}_ac.mp4".format(idx), data['images'])
    dir_name = "maze2d-%s-noisy" % args.maze if args.noisy else "maze2d"
    if args.batch_idx >= 0:
        dir_name = os.path.join(dir_name, "batch_{}".format(args.batch_idx))
    os.makedirs(os.path.join(args.data_dir, dir_name), exist_ok=True)
    file_name = os.path.join(args.data_dir, dir_name, "rollout_{}.h5".format(idx))

    # save rollout to file
    f = h5py.File(file_name, "w")
    f.create_dataset("traj_per_file", data=1)

    # store trajectory info in traj0 group
    npify(data)
    traj_data = f.create_group("traj0")
    traj_data.create_dataset("states", data=data["states"])
    traj_data.create_dataset("actions", data=data["actions"])

    terminals = data["terminals"]
    if np.sum(terminals) == 0:
        terminals[-1] = True

    # build pad-mask that indicates how long sequence is
    is_terminal_idxs = np.nonzero(terminals)[0]
    pad_mask = np.zeros((len(terminals),))
    pad_mask[: is_terminal_idxs[0]] = 1.0
    traj_data.create_dataset("pad_mask", data=pad_mask)

    f.close()


if __name__ == "__main__":
    main()
