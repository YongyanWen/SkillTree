from strl.utils.general_utils import AttrDict
from strl.data.kitchen.src.kitchen_data_loader import D4RLSequenceSplitDataset


data_spec = AttrDict(
    dataset_class=D4RLSequenceSplitDataset,
    n_actions=9,
    state_dim=60,
    env_name="kitchen-mkbl-v0",
    res=128,
    crop_rand_subseq=True,
)
data_spec.max_seq_len = 280
