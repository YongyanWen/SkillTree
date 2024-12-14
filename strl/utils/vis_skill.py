import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

K = 8

# plt.rcParams['text.usetex'] = True

N = 256
vals = np.ones((N, 3))
vals[:, 0] = np.linspace(204 / (N - 1), 247 / (N - 1), N)  # red
vals[:, 1] = np.linspace(231 / (N - 1), 202 / (N - 1), N)  # green
vals[:, 2] = np.linspace(255 / (N - 1), 57 / (N - 1), N)  # blue
cmap = ListedColormap(vals)


def get_filenames_recursive(directory):
    filenames = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            filenames.append(path)
        elif os.path.isdir(path):
            filenames.extend(get_filenames_recursive(path))
    return filenames


def plot_skill(file_path):
    dataset = h5py.File(file_path, 'r')

    skill = np.zeros((len(dataset['traj']['hl_action_index']), K))
    for i, s in enumerate(dataset['traj']['hl_action_index']):
        skill[i, s] = 1

    plt.figure(figsize=(4, 1.6))
    label_size = 5

    interval = 1

    x_labels = [i for i in range(0, len(dataset['traj']['hl_action_index']), interval)]
    y_labels = [i for i in range(K)]

    plt.yticks(np.arange(K), size=label_size)
    plt.xticks(np.arange(len(x_labels)) * interval, labels=x_labels, size=label_size)
    # plt.title("Skill Evaluation", size=label_size)

    # for i in range(K):
    #     for j in range(len(TASK_ELEMENTS)):
    #         plt.text(i, j, rate[i, j], ha="center", va="center", color="w", size=label_size)

    l_s = -0.5
    for i, step in enumerate(dataset['traj']['ct_step']):
        plt.plot([step / 10, step / 10], [-0.5, K - 0.5], color='orange', linewidth=0.5)
        plt.text((step + l_s) / 2 / 10, -1.2,
                 str(dataset['traj']['complete_task'][i][0])[2:-1], size=8, ha='center', va='center')
        l_s = step

    plt.xlabel('Time Step (× 10)', size=8)
    plt.ylabel('Skill Index', size=8)
    plt.imshow(skill.T, aspect='auto', cmap=cmap)
    # cb = plt.colorbar()
    # cb.ax.tick_params(labelsize=label_size)
    plt.tight_layout(pad=0.01)
    # plt.title('')
    # plt.show()
    plt.savefig(file_path + '.pdf')


if __name__ == '__main__':
    # file = '/home/wenyongyan/文档/spirl-master/spirl/experiments/hrl/kitchen/spirl_cl_vq/mkbl_s0_k16_inverse_kl/episode_3.h5'
    for i in range(10):
        # file = f'/home/wenyongyan/文档/spirl-master/spirl/experiments/hrl/kitchen/spirl_cl_vq/mkbl/10_{i+1}.h5'
        file = f'spirl/experiments/hrl/calvin/tree/test/fine_10_{i + 1}.h5'
        plot_skill(file)
