# SkillTree: Explainable Skill-Based Deep Reinforcement Learning for Long-Horizon Control Tasks

#### [[Paper]](https://arxiv.org/abs/2411.12173)

This is the official implementation of the paper "**SkillTree: Explainable Skill-Based Deep Reinforcement Learning for Long-Horizon Control Tasks**" (AAAI 2025). The codebase is built on top of [SPiRL](https://github.com/clvrai/spirl).

## Requirements

- python 3.10
- mujoco 2.1.0

## Installation Instructions

Create a virtual environment and install required packages.
```
pip3 install -r requirements.txt
pip3 install -e .
```

## Run SkillTree

To train a pre-trained skill model for the kitchen environment, run the following command:
```
python train.py --path=strl/configs/skill_prior_learning/kitchen/hierarchical_cl_vq_cdt --val_data_size=160
```
The results will be logged to [WandB](https://www.wandb.com/). Make sure to update the `WANDB_PROJECT_NAME` and `WANDB_ENTITY_NAME` in `strl.configs.local` with your project and entity names.

For training decision tree RL policy on the kitchen environment using the pre-trained skill, run:

```
python rl/train.py --path=strl/configs/hrl/kitchen/cdt_cl_vq_prior_cdt --seed=0 --prefix=s0
```

Before distilling the decision tree, collect the high-level decision tree policy dataset using the trained RL policy:

```
python strl/rl/dataset.py --path=strl/configs/hrl/kitchen/cdt_cl_vq_prior_cdt_eval
```

After obtaining the dataset, distill the hard decision tree using the following command:

```
python strl/decision_tree.py --path=strl/experiments/hrl/kitchen/cdt_cl_vq_prior_cdt_eval 
```

To evaluate the decision tree, run:

```
 python strl/rl/dhl_eval.py --path=strl/configs/hrl/kitchen/tree
```

## Run Baselines

- Train **SPiRL skill prior**:
```
python strl/train.py --path=strl/configs/skill_prior_learning/kitchen/spirl_cl --val_data_size=160
```

- Run **SPiRL**:
```
python strl/rl/train.py --path=strl/configs/hrl/kitchen/spirl_cl --seed=0 --prefix=s0
```

- Train **VQ-SPiRL skill prior**:

```
python strl/train.py --path=strl/configs/skill_prior_learning/kitchen/spirl_cl_vq --val_data_size=160
```

- Run **VQ-SPiRL**:
```
python strl/rl/train.py --path=strl/configs/hrl/kitchen/spirl_cl_vq/ --seed=0 --prefix=s0
```

## Hyperparameter Configuration

Hyperparameters can be modified in the `strl/config` directory. For example:

- **Skill Pretraining Hyperparameters**
  The hyperparameters for skill pretraining in the *kitchen* environment are located at:
  `strl/configs/skill_prior_learning/kitchen/hierarchical_cl_vq_cdt`
- **Reinforcement Learning Fine-Tuning Hyperparameters**
  The hyperparameters for RL fine-tuning in the *kitchen* environment can be found at:
  `strl/configs/hrl/kitchen/cdt_cl_vq_prior_cdt`

Each configuration file defines the parameters used for training, including learning rates, batch sizes, skill representations, and architecture details. To adjust these settings, edit the corresponding file directly or create a new configuration file tailored to your experiment.

## Environments and Datasets

- To switch between different kitchen tasks, modify the configuration at `strl/rl/envs/kitchen.py:13`.
- CALVIN can be downloaded from its official repository at https://github.com/mees/calvin.
- Download the CALVIN demonstration dataset from [this link](https://drive.google.com/file/d/1hR4mZ5AM1J_1za6TJL1BRok7J3XGCQGt/view?usp=sharing). Place the dataset file in the `strl/data/calvin` directory before starting the training process.

## Citation

If you find this codebase helpful for your research, please consider citing it as:

```
@misc{wen2024skilltree,
      title={SkillTree: Explainable Skill-Based Deep Reinforcement Learning for Long-Horizon Control Tasks}, 
      author={Yongyan Wen and Siyuan Li and Rongchang Zuo and Lei Yuan and Hangyu Mao and Peng Liu},
      year={2024},
      eprint={2411.12173},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2411.12173}, 
}
```
