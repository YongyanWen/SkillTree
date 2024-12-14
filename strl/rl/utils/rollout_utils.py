import os
import cv2
import h5py
import numpy as np


class RolloutSaver(object):
    """Saves rollout episodes to a target directory."""

    def __init__(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        self.counter = 0

    def save_rollout_to_file(self, episode):
        """Saves an episode to the next file index of the target folder."""
        # get save path
        save_path = os.path.join(self.save_dir, "rollout_{}.h5".format(self.counter))

        # save rollout to file
        f = h5py.File(save_path, "w")
        f.create_dataset("traj_per_file", data=1)

        # store trajectory info in traj0 group
        traj_data = f.create_group("traj")
        traj_data.create_dataset("states", data=np.array(episode.observation))
        # traj_data.create_dataset("images", data=np.array(episode.image, dtype=np.uint8))
        traj_data.create_dataset("actions", data=np.array(episode.action))

        terminals = np.array(episode.done)
        if np.sum(terminals) == 0:
            terminals[-1] = True

        # build pad-mask that indicates how long sequence is
        is_terminal_idxs = np.nonzero(terminals)[0]
        pad_mask = np.zeros((len(terminals),))
        pad_mask[:is_terminal_idxs[0]] = 1.
        traj_data.create_dataset("pad_mask", data=pad_mask)

        f.close()

        self.counter += 1

    def _resize_video(self, images, dim=64):
        """Resize a video in numpy array form to target dimension."""
        ret = np.zeros((images.shape[0], dim, dim, 3))

        for i in range(images.shape[0]):
            ret[i] = cv2.resize(images[i], dsize=(dim, dim),
                                interpolation=cv2.INTER_CUBIC)

        return ret.astype(np.uint8)

    def reset(self):
        """Resets counter."""
        self.counter = 0


class HPRolloutSaver(object):
    """Saves rollout episodes to a target directory."""

    def __init__(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        self.data = None
        self.num_episode = 0
        self.sample_hl = True

    def save_rollout(self, episode):
        # if self.data is None:
        # self.data = None

        if self.sample_hl:
            index = np.where(np.array(episode.is_hl_step))
            if self.data is None:
                self.data = {}
                self.data['observation'] = np.array(episode['observation'])[index]
                self.data['hl_action_index'] = np.array(episode['hl_action_index'])[index]
                # self.data['action'] = np.array(episode['action'])[index]
                # complete_task = []
                # ct_step = []
                # for i, t in enumerate(episode['info']):
                #     if len(t[0]['completed_task']) > 0:
                #         complete_task.append(list(t[0]['completed_task']))
                #         ct_step.append(i)
                # print(ct_step)
                # self.data['complete_task'] = complete_task
                # print(complete_task)
                # self.data['ct_step'] = ct_step
            else:
                self.data['observation'] = np.append(self.data['observation'], np.array(episode['observation'])[index],
                                                     axis=0)
                self.data['hl_action_index'] = np.append(self.data['hl_action_index'],
                                                         np.array(episode['hl_action_index'])[index], axis=0)
                # self.data['action'] = np.append(self.data['action'], np.array(episode['action'])[index], axis=0)
                # complete_task = []
                # ct_step = []
                # for i, t in enumerate(episode['info']):
                #     if len(t[0]['completed_task']) > 0:
                #         complete_task.append(list(t[0]['completed_task']))
                #         ct_step.append(i)
                # self.data['complete_task'].append(complete_task)
                # self.data['ct_step'].append(ct_step)
            self.num_episode += 1
        else:
            if self.data is None:
                self.data = {}
                self.data['observation'] = np.array(episode['observation'])
                self.data['action'] = np.array(episode['action'])
            else:
                self.data['observation'] = np.append(self.data['observation'], np.array(episode['observation']),
                                                     axis=0)
                self.data['action'] = np.append(self.data['action'], np.array(episode['action']), axis=0)
            self.num_episode += 1

    def save(self, file_name, save_interval=50):
        if self.num_episode > 0 and self.num_episode % save_interval == 0:
            save_path = os.path.join(self.save_dir, f"{file_name}_{self.num_episode}.h5")

            # save rollout to file
            f = h5py.File(save_path, "w")
            f.create_dataset("traj_per_file", data=1)

            # store trajectory info in traj group
            traj_data = f.create_group("traj")
            traj_data.create_dataset("states", data=self.data['observation'])
            # traj_data.create_dataset("actions", data=self.data['action'])
            traj_data.create_dataset("hl_action_index", data=self.data['hl_action_index'])
            # traj_data.create_dataset("complete_task", data=self.data['complete_task'])
            # traj_data.create_dataset("ct_step", data=self.data['ct_step'])

    def reset(self):
        """Resets counter."""
        self.num_episode = 0


class SERolloutSaver(object):
    """Saves rollout episodes to a target directory."""

    def __init__(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        self.data = None
        self.counter = 0

    def save_rollout(self, episode):
        # if self.data is None:
        self.data = {}
        self.data['observation'] = np.array(episode['observation'])
        # self.data['hl_action_index'] = np.array(episode['hl_action_index'])[index]

    def save(self, file_name):
        save_path = os.path.join(self.save_dir, f"rollout_{file_name}.h5")

        # save rollout to file
        f = h5py.File(save_path, "w")
        f.create_dataset("traj_per_file", data=1)

        # store trajectory info in traj0 group
        traj_data = f.create_group("traj")
        traj_data.create_dataset("states", data=self.data['observation'])
        # traj_data.create_dataset("actions", data=np.array(episode.action))
        # traj_data.create_dataset("hl_action_index", data=self.data['hl_action_index'])
        # self.counter += 1

    def reset(self):
        """Resets counter."""
        self.counter = 0
