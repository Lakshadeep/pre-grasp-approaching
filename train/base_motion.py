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

from base_pose_optimization.tasks.optimize_base_poses import OptimizeBasePoses

from mushroom_rl.algorithms.actor_critic import SAC, DDPG, SAC_hybrid, GaRrHyRL, BHyRL, PPO
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments.gym_env import Gym
from mushroom_rl.utils.dataset import compute_J, parse_dataset


from base_pose_optimization.networks.optimize_base_poses_mlp import ActorNetwork, CriticNetwork


def fill_replay_buffer_with_prior(cfg):
    data = np.load('{}'.format(cfg.task.train.replay_buffer_prior_path))
    
    states = data['states']
    next_states = data['states']
    actions = data['actions']
    time = data['rewards']
    
    n = len(states)

    # rewards = np.ones((n, )) * 100000
    rewards = (10000/(time+1)) + 10000
    absorbings = np.full((n, ), True)
    lasts = np.full((n, ), True)

    dataset = list()
    for s, a, t, r, ss, ab, last in zip(states, actions, time, rewards, next_states,
                                     absorbings.astype(bool), lasts.astype(bool)
                                     ):
        if t.item(0) > 0:
            dataset.append((s, a, r.item(0), ss, ab.item(0), last.item(0)))
    print("Replay buffer size with prior data:", len(dataset))
    # shuffle dataset
    random.shuffle(dataset)
    return dataset


def experiment(cfg, alg):
    np.random.seed()

    logger = Logger(alg.__name__, results_dir=cfg.task.train.save_dir)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    # MDP
    mdp = OptimizeBasePoses(cfg)

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
    target_entropy = cfg.task.train.target_entropy

    use_cuda = torch.cuda.is_available()

    # Approximator
    actor_input_shape = mdp.info.observation_space.shape

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

    agent = None

   
    # agent = alg(mdp.info, actor_mu_params, actor_sigma_params, actor_optimizer, critic_params, batch_size, 
    #     initial_replay_size, max_replay_size, warmup_transitions, tau, lr_alpha, target_entropy=target_entropy, critic_fit_params=None)

    agent = alg(mdp.info, actor_mu_params, actor_sigma_params, actor_optimizer, critic_params, batch_size, 
        initial_replay_size, max_replay_size, warmup_transitions, tau, lr_alpha, critic_fit_params=None)


    # agent_path = os.path.join(cfg.task.train.pre_trained_agent_dir, '{}.msh'.format(cfg.task.train.pre_trained_agent_name))
    # print("Agent path:", agent_path)
    # agent = alg.load(agent_path)

    if cfg.task.train.use_replay_buffer_prior:
        dataset = fill_replay_buffer_with_prior(cfg)
        agent.add_prior_data_to_replay_buffer(dataset)
    # mdp.shutdown()

    # Algorithm
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
            if n % 2 == 0:
                agent.save('{}/optimize_base_pose_{}_epoch_{}.msh'.format(save_dir, alg.__name__, n), full_save=True)
            J_mean_logs[n] = J_mean
            J_var_logs[n] = J_var
            R_mean_logs[n] = R_mean
            R_var_logs[n] = R_var
            E_logs[n] = E
            np.savez('{}/data_logs.npz'.format(save_dir), full_save=True, J=J_mean_logs, R=R_mean_logs, E=E_logs, J_var=J_var_logs, R_var=R_var_logs)

        # mdp.hard_reset()
        # print("Replay buffer size:", agent.get_replay_memory_size())
        # if cfg.task.train.use_replay_buffer_prior:
        #     dataset = fill_replay_buffer_with_prior(cfg)
        #     agent.add_prior_data_to_replay_buffer(dataset)

    
    agent.save('optimize_base_pose_{}.msh'.format(alg.__name__), full_save=True)
    np.savez('{}/data_logs.npz'.format(save_dir), full_save=True, J=J_mean_logs, R=R_mean_logs, E=E_logs, J_var=J_var_logs, R_var=R_var_logs)

    print("Training done")
    mdp.shutdown()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg : DictConfig) -> None:
    config = OmegaConf.to_yaml(cfg)
    print(config)
    experiment(cfg, alg=SAC)


if __name__ == '__main__':
    main()
