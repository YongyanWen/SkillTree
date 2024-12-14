import os

from strl.utils.general_utils import AttrDict
from strl.rl.agents.ac_agent import SACAgent
from strl.rl.policies.mlp_policies import MLPPolicy
from strl.rl.components.critic import MLPCritic
from strl.rl.components.replay_buffer import UniformReplayBuffer
from strl.rl.envs.calvin import CalvinEnv
from strl.rl.components.normalization import Normalizer
from strl.configs.default_data_configs.calvin import data_spec


current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'non-hierarchical RL experiments in kitchen env'

configuration = {
    'seed': 42,
    'agent': SACAgent,
    'environment': CalvinEnv,
    'data_dir': '.',
    'num_epochs': 21,
    'max_rollout_len': 360,
    'n_steps_per_epoch': 100000,
    'n_warmup_steps': 256,
}
configuration = AttrDict(configuration)

# Policy
policy_params = AttrDict(
    action_dim=data_spec.n_actions,
    input_dim=data_spec.state_dim,
    n_layers=5,      # number of policy network layers
    nz_mid=256,
    max_action_range=1.,
)

# Critic
critic_params = AttrDict(
    action_dim=policy_params.action_dim,
    input_dim=policy_params.input_dim,
    output_dim=1,
    n_layers=2,      # number of policy network layers
    nz_mid=256,
    action_input=True,
)

# Replay Buffer
replay_params = AttrDict(
    capacity=1e5,
    dump_replay=False,
)

# Agent
agent_config = AttrDict(
    policy=MLPPolicy,
    policy_params=policy_params,
    critic=MLPCritic,
    critic_params=critic_params,
    replay=UniformReplayBuffer,
    replay_params=replay_params,
    # obs_normalizer=Normalizer,
    # obs_normalizer_params=obs_norm_params,
    clip_q_target=False,
    batch_size=256,
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
              'turn_on_lightbulb': ['toggle_light', 'lightbulb', 0, 1],
              'move_slider_left': ['move_door_rel', 'base__slide', 0.15],  # 0 - 0.56
              'turn_on_led': ['toggle_light', 'led', 0, 1]
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