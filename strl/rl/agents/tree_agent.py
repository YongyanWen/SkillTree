import torch
import numpy as np

from strl.rl.components.agent import BaseAgent
from strl.utils.general_utils import ParamDict, ConstantSchedule, AttrDict
from strl.utils.pytorch_utils import check_shape, map2torch, map2np


class CARTAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self._hp = self._default_hparams().overwrite(config)
        self.policy = self._hp.policy(self._hp.policy_params)
        if not self.val_mode():
            self.oracle_policy = self._hp.policy_params.oracle_policy(self._hp.policy_params)
        self.replay_buffer = None

    def _default_hparams(self):
        default_dict = ParamDict({
        })
        return super()._default_hparams().overwrite(default_dict)

    def update(self, experience_batch):
        self.add_experience(experience_batch)
        output = self.policy.update(self.replay_buffer)
        return output

    def _act(self, obs, index=None, task=None):
        obs = map2torch(self._obs_normalizer(obs), self._hp.device)

        # if self.replay_buffer is None:
        #     if len(obs.shape) == 1:  # we need batched inputs for policy
        #         policy_output = self._remove_batch(self.oracle_policy(obs[None]))
        #         return map2np(policy_output)
        #     return map2np(self.oracle_policy(obs))

        if len(obs.shape) == 1:  # we need batched inputs for policy
            policy_output = self._remove_batch(self.policy(obs[None]))
            return map2np(policy_output)
        return map2np(self.policy(obs))

    def _act_rand(self, obs):
        policy_output = self.policy.sample_rand(map2torch(obs, self.policy.device))
        if 'dist' in policy_output:
            del policy_output['dist']
        return map2np(policy_output)

    def state_dict(self, *args, **kwargs):
        return self.policy.tree

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.policy_opt.load_state_dict(state_dict.pop('policy_opt'))
        super().load_state_dict(state_dict, *args, **kwargs)

    def visualize(self, logger, rollout_storage, step):
        super().visualize(logger, rollout_storage, step)
        self.policy.visualize(logger, rollout_storage, step)

    def add_experience(self, experience_batch):
        """Adds experience to replay buffer."""
        if not experience_batch:
            return  # pass if experience_batch is empty

        action_index = []
        for obs in experience_batch['observation']:
            oracle_output = map2np(self.oracle_policy(map2torch(obs, self._hp.device).unsqueeze(0)))
            action_index.append(oracle_output['action_index'])

        if self.replay_buffer is None:
            self.replay_buffer = {}
            self.replay_buffer['observation'] = experience_batch['observation']
            self.replay_buffer['hl_action_index'] = np.array(action_index)
        else:
            self.replay_buffer['observation'] = np.append(self.replay_buffer['observation'],
                                                          experience_batch['observation'],
                                                          axis=0)
            self.replay_buffer['hl_action_index'] = np.append(self.replay_buffer['hl_action_index'],
                                                              np.array(action_index),
                                                              axis=0)

    def reset(self):
        self.policy.reset()

    def _preprocess_experience(self, experience_batch):
        """Optionally pre-process experience before it is used for policy training."""
        return experience_batch