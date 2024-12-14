from strl.configs.hrl.calvin.spirl.conf import *
from strl.models.closed_loop_vq_cdt_mdl import ClVQCDTMdl
from strl.rl.policies.cl_model_policies import ClModelPolicy
from strl.rl.policies.prior_policies import LearnedVQPriorAugmentedPolicy, LearnedVQPriorAugmentedPolicyCDT

# update model params to conditioned decoder on state
ll_model_params.cond_decode = True

prior_model_name = "cdt_k8_s1_-1+15+6+0_4"
hl_model_name = "calvin_cdtk84_-1+15+6_n+b_s22_p0.5_1"

# CDT config
ll_model_params.update(AttrDict(
    codebook_K=8,
    fixed_codebook=False,
    feature_learning_depth=-1,
    num_intermediate_variables=15,
    decision_depth=6,
    greatest_path_probability=1,
    beta_fl=0,
    beta_dc=0,
    if_smooth=False,
    if_save=False,
    tree_name=""
    # if_freeze=False,
    # cdt_embedding_checkpoint=os.path.join(os.environ["EXP_DIR"], 
    #   f"skill_prior_learning/kitchen/hierarchical_cl_vq_cdt/{prior_model_name}/weights"),
))

# create LL closed-loop policy
ll_policy_params = AttrDict(
    policy_model=ClVQCDTMdl,
    policy_model_params=ll_model_params,
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                         f"skill_prior_learning/calvin/hierarchical_cl_vq_cdt/{prior_model_name}"),
    # policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"],
    #                                      "skill_prior_learning/kitchen/hierarchical_cl_vq/K_32"),
)
ll_policy_params.update(ll_model_params)

# create LL SAC agent (by default we will only use it for rolling out decoded skills, not finetuning skill decoder)
ll_agent_config = AttrDict(
    policy=ClModelPolicy,  # ClModelPolicy
    policy_params=ll_policy_params,
    critic=MLPCritic,  # LL critic is not used since we are not finetuning LL
    critic_params=hl_critic_params
)

hl_agent_config.policy = LearnedVQPriorAugmentedPolicyCDT

# update HL policy model params
hl_policy_params.update(AttrDict(
    policy=LearnedVQPriorAugmentedPolicy,  # PriorInitializedPolicy PriorAugmentedPolicy
    load_weights = True,
    policy_model_checkpoint = os.path.join(os.environ["EXP_DIR"], 
                                           f"hrl/calvin/cdt_cl_vq_prior_cdt/{hl_model_name}"),
    codebook_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                     f"hrl/calvin/cdt_cl_vq_prior_cdt/{hl_model_name}"),
    prior_model=ll_policy_params.policy_model,
    prior_model_params=ll_policy_params.policy_model_params,
    prior_model_checkpoint=ll_policy_params.policy_model_checkpoint,
    # policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"],
    #                                      "hrl/kitchen/spirl_cl_vq/mkbl_s0_k16_inverse_kl/weights"),
    squash_output_dist=False,
    policy_model_epoch=14,
))

# register new LL agent in agent_config and turn off LL agent updates
agent_config.update(AttrDict(
    hl_agent=ActionPriorSACAgent,
    hl_agent_params=hl_agent_config,
    ll_agent=SACAgent,
    ll_agent_params=ll_agent_config,
    update_ll=False,
))

agent_config.hl_agent_params.update(AttrDict(
    td_schedule_params=AttrDict(p=0.5),
))
