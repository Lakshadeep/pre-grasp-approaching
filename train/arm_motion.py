# python libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import trange
import sys
import random

import hydra
from omegaconf import DictConfig, OmegaConf
import os

from mushroom_rl.algorithms.actor_critic import SAC, DDPG, LRL, PPO
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments.gym_env import Gym
from mushroom_rl.utils.dataset import compute_J, parse_dataset

from pre_grasp_approaching.tasks.arm_motion import ArmMotion
from pre_grasp_approaching.networks.arm_motion import ActorNetwork, CriticNetwork


def experiment(cfg, alg):
    np.random.seed()

    logger = Logger(alg.__name__, results_dir=cfg.task.train.save_dir)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    
    # MDP
    mdp = ArmMotion(cfg.task)

    # Info
    print("MDP observation space low:", mdp.info.observation_space.low)
    print("MDP observation space high:", mdp.info.observation_space.high)

    # Settings
    initial_replay_size = cfg.task.train.initial_replay_size
    max_replay_size = cfg.task.train.max_replay_size
    batch_size = cfg.task.train.batch_size
    n_features = cfg.task.train.n_features
    warmup_transitions = cfg.task.train.warmup_transitions
    tau = cfg.task.train.tau
    lr_alpha = cfg.task.train.lr_alpha
    lr_actor = cfg.task.train.lr_actor
    lr_critic = cfg.task.train.lr_critic
    n_epochs = cfg.task.train.n_epochs

    use_cuda = torch.cuda.is_available()


    no_of_actions_from_prev_states = 4

    # Approximator
    actor_input_shape = (mdp.info.observation_space.shape[0] + no_of_actions_from_prev_states, )

    actor_mu_params = dict(network=ActorNetwork,
                           n_features=n_features,
                           input_shape=actor_input_shape,
                           output_shape=mdp.info.action_space.shape,
                           use_cuda=use_cuda)
    actor_sigma_params = dict(network=ActorNetwork,
                              n_features=n_features,
                              input_shape=actor_input_shape,
                              output_shape=mdp.info.action_space.shape,
                              use_cuda=use_cuda)

    action_space = mdp.info.action_space.shape


    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': lr_actor}}

    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': lr_critic}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         input_shape=critic_input_shape,
                         output_shape=(1,),
                         use_cuda=use_cuda)

    
    base_motion_agent = alg.load('{}/{}.msh'.format(cfg.task.train.base_motion_pre_trained_agent_dir, cfg.task.train.base_motion_pre_trained_agent_name))
    arm_motion_agent = alg.load('{}/{}.msh'.format(cfg.task.train.grasp_decision_pre_trained_agent_dir, cfg.task.train.grasp_decision_pre_trained_agent_name))

    agent = alg(mdp.info, [4,7], actor_mu_params, actor_sigma_params, None, actor_optimizer, critic_params, batch_size, 
        initial_replay_size, max_replay_size, warmup_transitions, tau, lr_alpha, critic_fit_params=None)

    agent.setup_prior([base_motion_agent, arm_motion_agent], np.array([[0,7],[0,7]]), np.array([[0,2],[2,4]]))
    core = Core(agent, mdp)


    # RUN
    dataset = core.evaluate(n_steps=cfg.task.train.n_steps_test, render=False)
    s, a, *_ = parse_dataset(dataset)


    J_mean = np.mean(compute_J(dataset, mdp.info.gamma))
    R_mean = np.mean(compute_J(dataset))

    E = agent.policy.entropy(s)

    logger.epoch_info(0, J=J_mean, R=R_mean, entropy=E)

    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    J_mean_logs = np.zeros(n_epochs,) 
    J_var_logs = np.zeros(n_epochs,) 
    R_mean_logs = np.zeros(n_epochs,)
    R_var_logs = np.zeros(n_epochs,)

    
    E_logs = np.zeros(n_epochs,) 

    for n in trange(n_epochs, leave=False):
        core.learn(n_steps=cfg.task.train.n_steps_train, n_steps_per_fit=1)
        dataset = core.evaluate(n_steps=cfg.task.train.n_steps_test, render=False)
        s, *_ = parse_dataset(dataset)

        J_mean = np.mean(compute_J(dataset, mdp.info.gamma))
        J_var = np.var(compute_J(dataset, mdp.info.gamma))
        R_mean = np.mean(compute_J(dataset))
        R_var = np.var(compute_J(dataset))

        E = agent.policy.entropy(s)

        logger.epoch_info(n+1, J=J_mean, R=R_mean, entropy=E)

        save_dir = cfg.task.train.save_dir
        if save_dir:
            if n % 10  == 0:
                agent.save('{}/arm_motion_epoch_{}.msh'.format(save_dir, n), full_save=True)
            
            J_mean_logs[n] = J_mean
            J_var_logs[n] = J_var
            R_mean_logs[n] = R_mean
            R_var_logs[n] = R_var
            E_logs[n] = E

        np.savez('{}/arm_motion_data_logs.npz'.format(save_dir), full_save=True, J=J_mean_logs, 
            R=R_mean_logs, E=E_logs, J_var=J_var_logs, R_var=R_var_logs)

    print("Training done")
    mdp.shutdown()


@hydra.main(version_base=None, config_path="../conf", config_name="arm_motion")
def main(cfg : DictConfig) -> None:
    config = OmegaConf.to_yaml(cfg)
    print(config)
    experiment(cfg, alg=LRL)


if __name__ == '__main__':
    main()