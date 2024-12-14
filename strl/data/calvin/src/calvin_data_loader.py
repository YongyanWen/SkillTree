import glob

import gym
import numpy as np
import os

from strl.components.data_loader import Dataset
from strl.utils.general_utils import AttrDict, shuffle_with_seed
import itertools


class CalvinSequenceSplitDataset(Dataset):
    SPLIT = AttrDict(train=0.99, val=0.01, test=0.0)

    def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1):
        self.phase = phase
        self.data_dir = data_dir
        self.spec = data_conf.dataset_spec
        self.subseq_len = self.spec.subseq_len
        self.remove_goal = self.spec.remove_goal if 'remove_goal' in self.spec else False
        self.dataset_size = dataset_size
        self.device = data_conf.device
        self.n_worker = 8
        self.shuffle = shuffle

        print('loading files from', self.data_dir)
        self.filenames = self._get_filenames()
        self.filenames = self._filter_filenames(self.filenames)
        self.dataset = self._get_samples_per_file(self.filenames[0])

        for data in self.dataset:
            data['obs'] = data['obs'][:, :21]

        # split dataset into sequences
        # seq_end_idxs = np.where(self.dataset['terminals'])[0]
        # start = 0
        self.seqs = []
        # for end_idx in seq_end_idxs:
        #     if end_idx + 1 - start < self.subseq_len: continue  # skip too short demos
        #     self.seqs.append(AttrDict(
        #         states=self.dataset['obs'][start:end_idx + 1],
        #         actions=self.dataset['actions'][start:end_idx + 1],
        #     ))
        #     start = end_idx +

        self.size = 0

        for data in self.dataset:
            self.seqs.append(AttrDict(
                states=data['obs'],
                actions=data['actions'],
                dones=data['dones'],
            ))
            self.size += len(data['obs'])

        # 0-pad sequences for skill-conditioned training
        if 'pad_n_steps' in self.spec and self.spec.pad_n_steps > 0:
            for seq in self.seqs:
                seq.states = np.concatenate(
                    (np.zeros((self.spec.pad_n_steps, seq.states.shape[1]), dtype=seq.states.dtype), seq.states))
                seq.actions = np.concatenate(
                    (np.zeros((self.spec.pad_n_steps, seq.actions.shape[1]), dtype=seq.actions.dtype), seq.actions))

        # filter demonstration sequences
        if 'filter_indices' in self.spec:
            print("!!! Filtering calvin demos in range {} !!!".format(self.spec.filter_indices))
            if not isinstance(self.spec.filter_indices[0], list):
                self.spec.filter_indices = [self.spec.filter_indices]
            self.seqs = list(itertools.chain.from_iterable([ \
                list(itertools.chain.from_iterable(itertools.repeat(x, self.spec.demo_repeats)
                                                   for x in self.seqs[fi[0]: fi[1] + 1])) for fi in
                self.spec.filter_indices]))
            import random
            random.shuffle(self.seqs)

        self.n_seqs = len(self.seqs)

        if self.phase == "train":
            self.start = 0
            self.end = int(self.SPLIT.train * self.n_seqs)
        elif self.phase == "val":
            self.start = int(self.SPLIT.train * self.n_seqs)
            self.end = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
        else:
            self.start = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
            self.end = self.n_seqs

    def _get_filenames(self):
        filenames = self._load_npz_files(self.data_dir)

        if not filenames:
            raise RuntimeError('No filenames found in {}'.format(self.data_dir))
        filenames = shuffle_with_seed(filenames)
        # filenames = self._split_with_percentage(self.spec.split, filenames)
        return filenames

    def _load_npz_files(self, dir):
        filenames = []
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(".npz"): filenames.append(os.path.join(root, file))
        return filenames

    def _get_samples_per_file(self, path):
        data = np.load(path, allow_pickle=True)
        return data

    def _split_with_percentage(self, frac, filenames):
        assert sum(frac.values()) <= 1.0  # fractions cannot sum up to more than 1
        assert self.phase in frac
        if self.phase == 'train':
            start, end = 0, frac['train']
        elif self.phase == 'val':
            start, end = frac['train'], frac['train'] + frac['val']
        else:
            start, end = frac['train'] + frac['val'], frac['train'] + frac['val'] + frac['mkbl']
        start, end = int(len(filenames) * start), int(len(filenames) * end)
        return filenames[start:end]

    def __getitem__(self, index):
        # sample start index in data range
        seq = self._sample_seq()
        start_idx = np.random.randint(0, seq.states.shape[0] - self.subseq_len - 1)
        output = AttrDict(
            states=seq.states[start_idx:start_idx + self.subseq_len],
            actions=seq.actions[start_idx:start_idx + self.subseq_len - 1],
            pad_mask=np.ones((self.subseq_len,)),
        )
        if self.remove_goal:
            output.states = output.states[..., :int(output.states.shape[-1] / 2)]
        return output

    def _sample_seq(self):
        return np.random.choice(self.seqs[self.start:self.end])

    def __len__(self):
        if self.dataset_size != -1:
            return self.dataset_size
        return int(self.SPLIT[self.phase] * self.size / self.subseq_len)
