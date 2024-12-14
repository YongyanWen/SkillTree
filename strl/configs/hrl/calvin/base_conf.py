import os
import copy

from strl.utils.general_utils import AttrDict
from strl.rl.components.agent import FixedIntervalHierarchicalAgent
from strl.rl.policies.mlp_policies import MLPPolicy
from strl.rl.components.critic import MLPCritic
from strl.rl.envs.calvin import CalvinEnv
from strl.rl.components.sampler import HierarchicalSampler
from strl.rl.components.replay_buffer import UniformReplayBuffer
from strl.rl.agents.ac_agent import SACAgent
from strl.rl.agents.skill_space_agent import SkillSpaceAgent
from strl.models.skill_prior_mdl import SkillPriorMdl
from strl.configs.default_data_configs.calvin import data_spec

current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'hierarchical RL on the calvin env'

configuration = {
    'seed': 42,
    'agent': FixedIntervalHierarchicalAgent,
    'environment': CalvinEnv,
    'sampler': HierarchicalSampler,
    'data_dir': '.',
    'num_epochs': 16,
    'max_rollout_len': 360,
    'n_steps_per_epoch': 100000,
    # 'n_warmup_steps': 5e3,
    'n_warmup_steps': 512,
}
configuration = AttrDict(configuration)

# Replay Buffer
replay_params = AttrDict(
)

# Observation Normalization
obs_norm_params = AttrDict(
)

base_agent_params = AttrDict(
    batch_size=256,
    replay=UniformReplayBuffer,
    replay_params=replay_params,
    # obs_normalizer=Normalizer,
    # obs_normalizer_params=obs_norm_params,
    clip_q_target=False,
)

###### Low-Level ######
# LL Policy
ll_model_params = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    kl_div_weight=5e-4,
    nz_enc=128,
    nz_mid=128,
    n_processing_layers=5,
    nz_vae=10,
    n_rollout_steps=10,
    codebook_K=16
)

# LL Agent
ll_agent_config = copy.deepcopy(base_agent_params)
ll_agent_config.update(AttrDict(
    model=SkillPriorMdl,
    model_params=ll_model_params,
    model_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                  "skill_prior_learning/calvin/hierarchical"),
))

###### High-Level ########
# HL Policy
hl_policy_params = AttrDict(
    action_dim=ll_model_params.nz_vae,  # z-dimension of the skill VAE
    input_dim=data_spec.state_dim,
    max_action_range=2.,  # prior is Gaussian with unit variance
    nz_mid=256,
    n_layers=5,
)

# HL Critic
hl_critic_params = AttrDict(
    action_dim=hl_policy_params.action_dim,
    input_dim=hl_policy_params.input_dim,
    output_dim=1,
    n_layers=5,  # number of policy network laye
    nz_mid=256,
    action_input=True,
)

# HL Agent
hl_agent_config = copy.deepcopy(base_agent_params)
hl_agent_config.update(AttrDict(
    policy=MLPPolicy,
    policy_params=hl_policy_params,
    critic=MLPCritic,
    critic_params=hl_critic_params,
))

##### Joint Agent #######
agent_config = AttrDict(
    hl_agent=SACAgent,
    hl_agent_params=hl_agent_config,
    ll_agent=SkillSpaceAgent,
    ll_agent_params=ll_agent_config,
    hl_interval=ll_model_params.n_rollout_steps,
    log_video_caption=True,
)

# Dataset - Random data
data_config = AttrDict()
data_config.dataset_spec = data_spec

# Environment
env_config = AttrDict(
    id='calvin',
    reward_norm=100,
    screen_size=[200, 200],
    action_repeat=1,
    frame_stack=1,
    absorbing_state=False,
    pixel_ob=False,
    state_ob=True,
    max_episode_steps=360,
    num_sequences=1000,

    data_path='/data',
    save_dir='/tmp',
    record=True,
    seed=0,
    bullet_time_step=240.0,
    use_vr=False,
    show_gui=False,
    use_scene_info=True,
    use_egl=False,
    control_freq=30,

    tasks={'_target_': 'calvin_env.envs.tasks.Tasks',
           'tasks': {
               'open_drawer': ['move_door_rel', 'base__drawer', 0.12],  # 0 - 0.24
               'turn_on_lightbulb': ['toggle_light', 'lightbulb', 0, 1],  # prob = Nan
               'move_slider_left': ['move_door_rel', 'base__slide', 0.15],  # 0 - 0.56, prob = 0.163%
               'turn_on_led': ['toggle_light', 'led', 0, 1]  # prob = 0.147%

               # 'open_drawer': ['move_door_rel', 'base__drawer', 0.12],  # 0 - 0.24, prob = 0.179%
               # 'move_slider_left': ['move_door_rel', 'base__slide', 0.15],  # 0 - 0.56, prob = 0.130%
               # 'turn_on_led': ['toggle_light', 'led', 0, 1],
               # 'turn_on_lightbulb': ['toggle_light', 'lightbulb', 0, 1],  # prob = 0.244%
           },
           },

    # camera
    cameras={
        'static': {
            '_target_': 'calvin_env.camera.static_camera.StaticCamera',
            'name': 'static',
            'fov': 10,
            'aspect': 1,
            'nearval': 0.01,
            'farval': 10,
            'width': 200,
            'height': 200,
            'look_at': [-0.026242351159453392, -0.0302329882979393, 0.3920000493526459],
            'look_from': [2.871459009488717, -2.166602199425597, 2.555159848480571],
            'up_vector': [0.4041403970338857, 0.22629790978217404, 0.8862616969685161],
        }
    },

    # scene
    scene_cfg={
        '_target_': 'calvin_env.scene.play_table_scene.PlayTableScene',
        '_recursive_': False,
        'data_path': 'data',  ## ${data_path}
        'global_scaling': 0.8,
        'euler_obs': True,  ## ${robot_cfg.euler_obs}
        'robot_base_position': [-0.34, -0.46, 0.24],
        'robot_base_orientation': [0, 0, 0],
        'robot_initial_joint_positions': [-1.21779206, 1.03987646, 2.11978261, -2.34205014, -0.87015947, 1.64119353,
                                          0.55344866],
        'surfaces': {
            'table': [[0.0, -0.15, 0.46], [0.35, -0.03, 0.46]],
            'slider_left': [[-0.32, 0.05, 0.46], [-0.16, 0.12, 0.46]],
            'slider_right': [[-0.05, 0.05, 0.46], [0.13, 0.12, 0.46]]
        },
        'objects': {
            'fixed_objects': {
                'table': {
                    'file': 'calvin_table_D/urdf/calvin_table_D.urdf',
                    'initial_pos': [0, 0, 0],
                    'initial_orn': [0, 0, 0],
                    'joints': {
                        'base__slide': {
                            'initial_state': 0},  # Prismatic
                        'base__drawer': {
                            'initial_state': 0},  # Prismatic
                    },
                    'buttons': {
                        'base__button': {
                            'initial_state': 0,  # Prismatic
                            'effect': 'led'}
                    },
                    'switches': {
                        'base__switch': {
                            'initial_state': 0,  # Revolute
                            'effect': 'lightbulb',
                        }
                    },
                    'lights': {
                        'lightbulb': {
                            'link': 'light_link',
                            'color': [1, 1, 0, 1]
                        },  # yellow
                        'led': {
                            'link': 'led_link',
                            'color': [0, 1, 0, 1]
                        }
                    }  # green
                }
            },
            'movable_objects': {
                'block_red': {
                    'file': 'blocks/block_red_middle.urdf',
                    'initial_pos': 'any',
                    'initial_orn': 'any'
                },
                'block_blue': {
                    'file': 'blocks/block_blue_small.urdf',
                    'initial_pos': 'any',
                    'initial_orn': 'any'
                },
                'block_pink': {
                    'file': 'blocks/block_pink_big.urdf',
                    'initial_pos': 'any',
                    'initial_orn': 'any'
                }
            }
        }
    },

    # robot
    robot_cfg={
        '_target_': 'calvin_env.robot.robot.Robot',
        'filename': 'franka_panda/panda_longer_finger.urdf',
        'base_position': [-0.34, -0.46, 0.24],  ## ${scene.robot_base_position}
        'base_orientation': [0, 0, 0],  ## ${scene.robot_base_orientation}
        'initial_joint_positions': [-1.21779206, 1.03987646, 2.11978261, -2.34205014, -0.87015947, 1.64119353,
                                    0.55344866],  ## ${scene.robot_initial_joint_positions}
        'max_joint_force': 200.0,
        'gripper_force': 200,
        'arm_joint_ids': [0, 1, 2, 3, 4, 5, 6],
        'lower_joint_limits': [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
        'upper_joint_limits': [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
        'gripper_joint_ids': [9, 11],
        'gripper_joint_limits': [0, 0.04],
        'tcp_link_id': 15,
        'end_effector_link_id': 7,
        'gripper_cam_link': 12,
        'use_nullspace': True,
        'max_velocity': 2,
        'use_ik_fast': False,
        'magic_scaling_factor_pos': 1,  # 1.6
        'magic_scaling_factor_orn': 1,  # 2.2
        'use_target_pose': True,
        'euler_obs': True,
    }
)
