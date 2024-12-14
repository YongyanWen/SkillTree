from strl.configs.hrl.kitchen.spirl.conf import *
from strl.models.skill_prior_vq_mdl import SkillPriorVQMdl
from strl.rl.policies.cl_model_policies import ClModelPolicy
from strl.rl.policies.prior_policies import LearnedVQPriorAugmentedPolicy

# update model params to conditioned decoder on state
ll_model_params.cond_decode = False

ll_model_params.update(AttrDict(
    codebook_K=16,
))

# create LL closed-loop policy
ll_policy_params = AttrDict(
    policy_model=SkillPriorVQMdl,
    policy_model_params=ll_model_params,
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                         "skill_prior_learning/kitchen/hierarchical_vq/K_16"),
)
ll_policy_params.update(ll_model_params)

ll_agent_config.update(AttrDict(
    model=SkillPriorVQMdl,
    model_params=ll_model_params,
    model_checkpoint=ll_policy_params.policy_model_checkpoint,
))

hl_agent_config.policy = LearnedVQPriorAugmentedPolicy

# update HL policy model params
hl_policy_params.update(AttrDict(
    policy=LearnedVQPriorAugmentedPolicy,  # PriorInitializedPolicy PriorAugmentedPolicy
    prior_model=ll_policy_params.policy_model,
    prior_model_params=ll_policy_params.policy_model_params,
    prior_model_checkpoint=ll_policy_params.policy_model_checkpoint,
))

# register new LL agent in agent_config and turn off LL agent updates
agent_config.update(AttrDict(
    hl_agent=ActionPriorSACAgent,
    hl_agent_params=hl_agent_config,
    ll_agent=SkillSpaceAgent,
    ll_agent_params=ll_agent_config,
    update_ll=False,
))
