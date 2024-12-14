import torch
import torch.nn.functional as F
import math
import numpy as np

from torch.distributions.multivariate_normal import MultivariateNormal

from strl.utils.pytorch_utils import ten2ar
from strl.utils.general_utils import batch_apply
from scipy.stats import wasserstein_distance

class Categorical:
    """ Represents a categorical distribution """

    # TODO: implement a dict conversion function
    def __init__(self, probs=None, logits=None, codebook=None, fixed=False):
        self.prob = torch.distributions.Categorical(probs=probs, logits=logits)
        self.codebook = codebook

        if codebook is not None:
            self.codebook.embedding.weight.requires_grad = not fixed

    def sample(self):
        if self.codebook is not None:
            index = self.prob.sample()
            action = self.codebook.embedding.weight[index]
            log_prob = self.prob.log_prob(index)
            return action, index, log_prob
        else:
            return self.prob.sample()

    def kl_divergence(self, other):
        """Here self=q and other=p and we compute KL(q, p)"""
        delta = 1e-10
        # log_q = torch.log(self.prob.probs + delta)
        log_p = torch.log(other.prob.probs + delta)

        # return torch.sum(self.prob.probs * torch.log((self.prob.probs + delta) / (other.prob.probs + delta)), dim=-1)
        return torch.nn.functional.kl_div(log_p, self.prob.logits, reduction='none', log_target=True)

    def wasserstein_distance(self, other):
        u = self.prob.probs.cpu().detach().numpy()
        v = other.prob.probs.cpu().detach().numpy()
        if u.shape[0] == 1:
            return torch.from_numpy(wasserstein_distance(u, v)).float()
        else:
            d = torch.from_numpy(np.array([wasserstein_distance(u[i], v[i]) for i in range(u.shape[0])])).float()
            return d

    def nll(self, x):
        # Negative log likelihood (probability)
        return -1 * self.log_prob(x)

    def log_prob(self, val):
        if isinstance(val, tuple):
            val = val[1]
        return self.prob.log_prob(val)

    def entropy(self):
        return self.prob.entropy()

    @property
    def shape(self):
        return self.prob.probs.shape

    def rsample(self):
        """Identical to self.sample(), to conform with pytorch naming scheme."""
        return self.sample()

    def detach(self):
        """Detaches internal variables. Returns detached Gaussian."""
        return type(self)(logits=self.prob.logits.detach(), codebook=self.codebook)

    def to_numpy(self):
        """Convert internal variables to numpy arrays."""
        return self