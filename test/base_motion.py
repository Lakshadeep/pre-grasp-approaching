# python libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import trange

import hydra
from omegaconf import DictConfig, OmegaConf

from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments.gym_env import Gym
from mushroom_rl.utils.dataset import compute_J, parse_dataset


from pre_grasp_approaching.tasks.base_motion import BaseMotion


def experiment(cfg):
    np.random.seed()

    # MDP
    mdp = BaseMotion(cfg)

    # Info
    print("MDP observation space low:", mdp.info.observation_space.low)
    print("MDP observation space high:", mdp.info.observation_space.high)

    # Agent
    agent = alg.load('{}/{}.msh'.format(cfg.task.test.save_dir, cfg.task.test.agent_name))

    # Algorithm
    core = Core(agent, mdp)

    core.evaluate(n_episodes=cfg.task.test.n_episodes, render=False)
    # core.evaluate(n_steps=cfg.test.n_steps, render=False)

    print("Done!!")


@hydra.main(version_base=None, config_path="../conf", config_name="base_motion")
def main(cfg : DictConfig) -> None:
    config = OmegaConf.to_yaml(cfg)
    print(config)
    experiment(cfg)


if __name__ == '__main__':
    main()