import torch
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from strl.rl.components.policy import Policy
from strl.utils.general_utils import AttrDict, ParamDict


class DQNPolicy(Policy):
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        super().__init__()

        self.tau = self._hp.tau
        self.epsilon = self._hp.epsilon
        self.eps_min = self._hp.eps_end
        self.eps_decay = self._hp.eps_decay
        self.batch_size = self._hp.batch_size
        self.update_count = 0
        self.action_space = [i for i in range(self._hp.action_dim)]

        self.q_eval = DuelingDeepQNetwork(config=self._hp)
        self.q_target = DuelingDeepQNetwork(config=self._hp)

        # self.memory = ReplayBuffer(state_dim=state_dim, task_dim=task_dim, action_dim=action_dim,
        #                            max_size=self.max_size, batch_size=self.batch_size)

        self.update_network_parameters(tau=1.0)
        self.last_q_params = self.q_eval.parameters()

        self.codebook = self._load_codebook()

    def _default_hparams(self):
        default_dict = ParamDict({
            'tau': 1.0,
            'epsilon': 1.0,
            'eps_end': 0.01,
            'eps_decay': 0.995,
            'max_size': 50000,
            'target_update_interval': 16384,
            'batch_size': 256,
            'hidden_layers': [64, 64],
        })
        return super()._default_hparams().overwrite(default_dict)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for q_target_params, q_eval_params in zip(self.q_target.parameters(), self.q_eval.parameters()):
            q_target_params.data.copy_(tau * q_eval_params + (1 - tau) * q_target_params)

        # print('update network parameters')

    def decrement_epsilon(self):
        self.epsilon = self.epsilon * self.eps_decay if self.epsilon > self.eps_min else self.eps_min

    def save_parameters(self):
        self.last_q_params = self.q_eval.parameters()

    def _build_network(self):
        pass

    def recover(self):
        for last_q_params, q_eval_params in zip(self.last_q_params, self.q_eval.parameters()):
            q_eval_params.data.copy_(last_q_params)

    def forward(self, obs):
        q_vals = self.q_eval.forward(obs)
        action = T.argmax(q_vals, dim=-1)

        size = action.shape[0] if action.dim() > 0 else 1
        index = torch.arange(size, dtype=torch.long).to(self.device)
        if np.random.random() < self.epsilon:
            action = torch.from_numpy(np.array(np.random.choice(self._hp.codebook_K, size))).long()
        return AttrDict(action=self.codebook[action], action_index=action, value=q_vals[index, action])

    def sample_rand(self, obs, prior=False):
        if prior:
            return super().forward(obs)
        return self.forward(obs)

    def _load_codebook(self):
        weight = torch.load(self._hp.codebook_checkpoint)

        return weight['state_dict']['codebook.embedding.weight']
        # return weight['state_dict']['hl_agent']['policy.net.codebook.embedding.weight']


class DuelingDeepQNetwork(nn.Module):
    def __init__(self, config):
        super(DuelingDeepQNetwork, self).__init__()
        self.config = config
        self.fc_layers, self.V, self.A = self._build_net(self.config)

        # self.cnn = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=0),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(16, 64),
        #     nn.ReLU(),
        # )

    def forward(self, obs):
        x = obs
        #
        for i in range(len(self.fc_layers)):
            x = T.relu(self.fc_layers[i](x))

        v = self.V(x)
        a = self.A(x)
        q = v + a - T.mean(a, dim=-1, keepdim=True)

        return q

    def _build_net(self, config):
        fc_layers = nn.ModuleList([])

        curr_input_dim = config.input_dim
        for i in range(len(config.hidden_layers)):
            fc_layers.append(nn.Linear(curr_input_dim, config.hidden_layers[i]))
            curr_input_dim = config.hidden_layers[i]

        V = nn.Linear(curr_input_dim, 1)
        A = nn.Linear(curr_input_dim, config.codebook_K)

        return fc_layers, V, A

    # def save_checkpoint(self, checkpoint_file):
    #     T.save(self.state_dict(), checkpoint_file)
    #
    # def load_checkpoint(self, checkpoint_file):
    #     self.load_state_dict(T.load(checkpoint_file))

# class ReplayBuffer:
#     def __init__(self, state_dim, task_dim, action_dim, max_size, batch_size):
#         self.mem_size = max_size
#         self.batch_size = batch_size
#         self.mem_cnt = 0
#
#         self.state_memory = np.zeros((self.mem_size, *state_dim))
#         self.task_memory = np.zeros((self.mem_size, task_dim))
#         self.action_memory = np.zeros((self.mem_size,))
#         self.reward_memory = np.zeros((self.mem_size,))
#         self.next_state_memory = np.zeros((self.mem_size, *state_dim))
#         self.terminal_memory = np.zeros((self.mem_size,), dtype=bool)
#
#     def insert(self, state, task, action, reward, state_, done):
#         mem_idx = self.mem_cnt % self.mem_size
#
#         self.state_memory[mem_idx] = state.cpu()
#         self.task_memory[mem_idx] = task.cpu()
#         self.action_memory[mem_idx] = action.cpu()
#         self.reward_memory[mem_idx] = reward.cpu()
#         self.next_state_memory[mem_idx] = state_.cpu()
#         self.terminal_memory[mem_idx] = done
#
#         self.mem_cnt += 1
#
#     def sample_buffer(self):
#         mem_len = min(self.mem_size, self.mem_cnt)
#
#         batch = np.random.choice(mem_len, self.batch_size, replace=False)
#
#         states = self.state_memory[batch]
#         tasks = self.task_memory[batch]
#         actions = self.action_memory[batch]
#         rewards = self.reward_memory[batch]
#         states_ = self.next_state_memory[batch]
#         terminals = self.terminal_memory[batch]
#
#         return states, tasks, actions, rewards, states_, terminals
#
#     def ready(self):
#         return self.mem_cnt > self.batch_size