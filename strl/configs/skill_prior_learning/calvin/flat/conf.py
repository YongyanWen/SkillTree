import os

from strl.models.bc_mdl import BCMdl
from strl.components.logger import Logger
from strl.utils.general_utils import AttrDict
from strl.configs.default_data_configs.calvin import data_spec
from strl.components.evaluator import DummyEvaluator


current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': BCMdl,
    'logger': Logger,
    'data_dir': '.',
    'epoch_cycles_train': 10,
    'evaluator': DummyEvaluator,
}
configuration = AttrDict(configuration)

model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.subseq_len = 1 + 1  # flat last action from seq gets cropped
