import copy

from strl.configs.rl.calvin.base_conf import *
from strl.models.bc_mdl import BCMdl

prior_model_params = AttrDict(state_dim=data_spec.state_dim,
                              action_dim=data_spec.n_actions,
                              )

policy_params.update(AttrDict(
    prior_model=BCMdl,
    prior_model_params=copy.deepcopy(prior_model_params),
    prior_model_checkpoint=os.path.join(os.environ["EXP_DIR"], "skill_prior_learning/calvin/flat"),
    squash_output_dist=False,
    policy_model=BCMdl,
    policy_model_params=copy.deepcopy(prior_model_params),
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                         "rl/calvin/prior_initialized/bc_finetune/s3"),
    policy_model_epoch=14,
))