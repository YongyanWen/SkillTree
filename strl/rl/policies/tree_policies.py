import json

import torch
import numpy as np

from strl.utils.general_utils import AttrDict, ParamDict
from strl.rl.components.agent import BaseAgent
from strl.rl.components.policy import Policy
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree

import pickle


class CARTPolicy(Policy):
    def __init__(self, config):
        super().__init__()
        self._hp = self._default_hparams().overwrite(config)
        self.tree = self._load_tree()
        if self._hp.is_index:
            self.codebook = self._load_codebook()

    def _default_hparams(self):
        default_dict = ParamDict({
            'load_weights': True,  # optionally allows to *not* load the weights (ie train from scratch)
            'max_depth': 9,
            'is_index': True,
            'type': 'classifier'
        })
        return super()._default_hparams().overwrite(default_dict)

    def forward(self, obs):
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        if self._hp.is_index:
            action_index = self.tree.predict(obs)
            return AttrDict(action=self.codebook[action_index], action_index=action_index)
        else:
            return AttrDict(action=self.tree.predict(obs))

    def update(self, experience):
        self.tree.fit(experience['observation'], experience['hl_action_index'])

        info = AttrDict(
            leaf=self.tree.get_n_leaves(),
            depth=self.tree.get_depth(),
            node_count=self.tree.tree_.node_count,
            # importances=self.tree.feature_importances_,
            # gini=self.tree.tree_.impurity,
            test_score=self.tree.score(experience['observation'], experience['hl_action_index']),
        )
        return info

    def _build_network(self):
        pass

    def _load_tree(self):
        if self._hp.load_weights:
            with open(self._hp.policy_model_checkpoint, 'rb') as f:
                print('loading weights from {}'.format(self._hp.policy_model_checkpoint))
                tree = pickle.load(f)
            return tree

    def _load_codebook(self):
        weight = torch.load(self._hp.codebook_checkpoint)
        print('loading codebook from {}'.format(self._hp.codebook_checkpoint))

        # return weight['state_dict']['hl_agent']['policy.prior_net.codebook.embedding.weight']
        return weight['state_dict']['hl_agent']['policy.net.codebook.embedding.weight']

    @property
    def has_trainable_params(self):
        """Indicates whether policy has trainable params."""
        return False
