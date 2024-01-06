# python libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import trange

import hydra
from omegaconf import DictConfig, OmegaConf



from base_pose_optimization.tasks.optimize_base_poses import OptimizeBasePoses

from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments.gym_env import Gym
from mushroom_rl.utils.dataset import compute_J, parse_dataset


from base_pose_optimization.networks.optimize_base_poses_gnn_hetero import ActorNetwork, CriticNetwork


def experiment(cfg, alg):
    np.random.seed()

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    # MDP
    mdp = OptimizeBasePoses(cfg)

    # Info
    print("MDP observation space low:", mdp.info.observation_space.low)
    print("MDP observation space high:", mdp.info.observation_space.high)

    # Agent
    print("Agent location:", '{}/optimize_base_pose_{}_epoch_{}.msh'.format(cfg.task.test.save_dir, alg.__name__, cfg.task.test.epoch_no))
    agent = alg.load('{}/optimize_base_pose_{}_epoch_{}.msh'.format(cfg.task.test.save_dir, alg.__name__, cfg.task.test.epoch_no))

    # Algorithm
    core = Core(agent, mdp)

    core.evaluate(n_episodes=cfg.task.test.n_episodes, render=False)
    # core.evaluate(n_steps=cfg.test.n_steps, render=False)

    print("Done!!")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg : DictConfig) -> None:
    config = OmegaConf.to_yaml(cfg)
    print(config)
    experiment(cfg, alg=SAC)


if __name__ == '__main__':
    main()