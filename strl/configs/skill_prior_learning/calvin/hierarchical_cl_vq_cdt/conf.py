import os

from strl.models.closed_loop_vq_cdt_mdl import ClVQCDTMdl
from strl.components.logger import Logger
from strl.utils.general_utils import AttrDict
from strl.configs.default_data_configs.calvin import data_spec
from strl.components.evaluator import TopOfNSequenceEvaluator

current_dir = os.path.dirname(os.path.realpath(__file__))

configuration = {
    'model': ClVQCDTMdl,
    'logger': Logger,
    'data_dir': os.path.join(os.environ['DATA_DIR'], 'calvin'),
    'epoch_cycles_train': 50,
    'num_epochs': 100,
    'evaluator': TopOfNSequenceEvaluator,
    'top_of_n_eval': 100,
    'top_comp_metric': 'mse',
}
configuration = AttrDict(configuration)

model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    n_rollout_steps=10,
    nz_enc=128,
    nz_mid=128,
    n_processing_layers=5,
    cond_decode=True,
    codebook_K=64,
    commitment_beta=0.25,
    fixed_codebook=False,

    feature_learning_depth=-1,
    num_intermediate_variables=20,
    decision_depth=3,
    greatest_path_probability=0,
    beta_fl=0,
    beta_dc=0,
    if_smooth=False,
    if_save=False,
    tree_name="skill"
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.subseq_len = model_config.n_rollout_steps + 1  # flat last action from seq gets cropped
