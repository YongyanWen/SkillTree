from strl.configs.rl.calvin.prior_initialized.base_conf import *
from strl.rl.policies.prior_policies import PriorInitializedPolicy

agent_config.policy = PriorInitializedPolicy
configuration.agent = SACAgent