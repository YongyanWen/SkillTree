import torch
import numpy as np

from strl.rl.agents.discrete_ac_agent import DiscreteSACAgent
from strl.utils.general_utils import ParamDict, ConstantSchedule, AttrDict
from strl.utils.pytorch_utils import check_shape, map2torch


class ActionPriorDiscreteSACAgent(DiscreteSACAgent):
    """Implements SAC with non-uniform, learned action / skill prior."""

    def __init__(self, config):
        DiscreteSACAgent.__init__(self, config)
        self._target_divergence = self._hp.td_schedule(self._hp.td_schedule_params)
        # self.warm_step = 0

    def _default_hparams(self):
        default_dict = ParamDict({
            'alpha_min': None,  # minimum value alpha is clipped to, no clipping if None
            'alpha_max': 100,
            'td_schedule': ConstantSchedule,  # schedule used for target divergence param
            'td_schedule_params': AttrDict(  # parameters for target divergence schedule
                p=1.0,
            ),
        })
        return super()._default_hparams().overwrite(default_dict)

    def update(self, experience_batch):
        info = super().update(experience_batch)
        info.target_divergence = self._target_divergence(self.schedule_steps)
        return info

    def _compute_alpha_loss(self, policy_output):
        """Computes loss for alpha update based on target divergence."""
        return self.alpha * (policy_output.dist.prob.probs * (
                self._target_divergence(self.schedule_steps) - policy_output.prior_divergence[:, None])).detach().mean()

    def _compute_policy_loss(self, experience_batch, policy_output):
        """Computes loss for policy update."""
        q_est = torch.min(*[critic(experience_batch.observation).q
                            for critic in self.critics])
        # if self.warm_step < 1000000:
        #     policy_loss = -1 * q_est
        #     self.warm_step = self.warm_step + 1
        # else:
        #     policy_loss = -1 * q_est + self.alpha * policy_output.prior_divergence[:, None]
        policy_loss = policy_output.dist.prob.probs * (
                -1 * q_est + self.alpha * policy_output.prior_divergence[:, None])
        check_shape(policy_loss, [self._hp.batch_size, self._hp.critic_params.output_dim])
        return policy_loss.mean()

    def _compute_next_value(self, experience_batch, policy_output):
        """Computes value of next state for target value computation."""
        q_next = torch.min(*[critic_target(experience_batch.observation_next, self._prep_action(policy_output.action)).q
                             for critic_target in self.critic_targets])
        next_val = policy_output.dist.prob.probs * (q_next - self.alpha * policy_output.prior_divergence[:, None])
        check_shape(next_val, [self._hp.batch_size, self._hp.critic_params.output_dim])
        return next_val.squeeze(-1)

    def _aux_info(self, experience_batch, policy_output):
        """Stores any additional values that should get logged to WandB."""
        aux_info = super()._aux_info(experience_batch, policy_output)
        aux_info.prior_divergence = policy_output.prior_divergence.mean()
        if 'ensemble_divergence' in policy_output:  # when using ensemble thresholded prior divergence
            aux_info.ensemble_divergence = policy_output.ensemble_divergence.mean()
            aux_info.learned_prior_divergence = policy_output.learned_prior_divergence.mean()
            aux_info.below_ensemble_div_thresh = policy_output.below_ensemble_div_thresh.mean()
        return aux_info

    def state_dict(self, *args, **kwargs):
        d = super().state_dict(*args, **kwargs)
        d['update_steps'] = self._update_steps
        return d

    def load_state_dict(self, state_dict, *args, **kwargs):
        self._update_steps = state_dict.pop('update_steps')
        super().load_state_dict(state_dict, *args, **kwargs)

    @property
    def alpha(self):
        if self._hp.alpha_min is not None:
            return torch.clamp(super().alpha, min=self._hp.alpha_min)
        return super().alpha
