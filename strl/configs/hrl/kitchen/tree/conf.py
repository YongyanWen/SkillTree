from strl.configs.hrl.kitchen.spirl.conf import *
from strl.models.closed_loop_vq_cdt_mdl import ClVQCDTMdl
from strl.rl.policies.cl_model_policies import ClModelPolicy
from strl.rl.policies.prior_policies import LearnedVQPriorAugmentedPolicy
from strl.rl.policies.tree_policies import CARTPolicy
from strl.rl.agents.tree_agent import CARTAgent

# update model params to conditioned decoder on state
ll_model_params.cond_decode = True

ll_model_params.update(AttrDict(
    codebook_K=16,
    fixed_codebook=False,
    feature_learning_depth=-1,
    num_intermediate_variables=20,
    decision_depth=6,
    greatest_path_probability=1,
    beta_fl=0,
    beta_dc=0,
    if_smooth=False,
    if_save=False,
    tree_name=""
))

# create LL closed-loop policy
ll_policy_params = AttrDict(
    policy_model=ClVQCDTMdl,
    policy_model_params=ll_model_params,
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                         "skill_prior_learning/kitchen/hierarchical_cl_vq_cdt"),
)
ll_policy_params.update(ll_model_params)

# create LL SAC agent (by default we will only use it for rolling out decoded skills, not finetuning skill decoder)
ll_agent_config = AttrDict(
    policy=ClModelPolicy,
    policy_params=ll_policy_params,
    critic=MLPCritic,  # LL critic is not used since we are not finetuning LL
    critic_params=hl_critic_params
)

hl_agent_config.policy = CARTPolicy

# update HL policy model params
hl_policy_params.update(AttrDict(
    policy=CARTPolicy,
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                         "hrl/kitchen/cdt_cl_vq_prior_cdt_eval/cart_fine_100_d6.pkl"),
    codebook_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                     "hrl/kitchen/cdt_cl_vq_prior_cdt/s0/weights/weights_ep19.pth"),
    prior_model=ll_policy_params.policy_model,
    prior_model_params=ll_policy_params.policy_model_params,
    prior_model_checkpoint=ll_policy_params.policy_model_checkpoint,
))

# register new LL agent in agent_config and turn off LL agent updates
agent_config.update(AttrDict(
    hl_agent=CARTAgent,
    hl_agent_params=hl_agent_config,
    ll_agent=SACAgent,
    ll_agent_params=ll_agent_config,
    update_ll=False,
))
