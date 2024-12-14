from strl.data.calvin.src.calvin_data_loader import CalvinSequenceSplitDataset
from strl.utils.general_utils import AttrDict

data_spec = AttrDict(
    dataset_class=CalvinSequenceSplitDataset,
    n_actions=7,
    state_dim=21,
    # env_name="calvin",
    res=64,
    use_rel_action=1,
    split=AttrDict(train=0.99, val=0.01, test=0.0),
    crop_rand_subseq=True,
)
data_spec.max_seq_len = 360
