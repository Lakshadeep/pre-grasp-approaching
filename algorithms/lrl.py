import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DeepAC
from mushroom_rl.policy import Policy
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.replay_memory import ReplayMemory
from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl.utils.parameters import to_parameter

from copy import deepcopy
from itertools import chain


class GumbelSoftmax(torch.distributions.RelaxedOneHotCategorical):
    '''
    A differentiable Categorical distribution using reparametrization trick with Gumbel-Softmax
    Explanation http://amid.fish/assets/gumbel.html
    NOTE: use this in place PyTorch's RelaxedOneHotCategorical distribution since its log_prob is not working right (returns positive values)
    Papers:
    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables (Maddison et al, 2017)
    [2] Categorical Reparametrization with Gumbel-Softmax (Jang et al, 2017)
    '''

    def sample(self, sample_shape=torch.Size()):
        '''Gumbel-softmax sampling. Note rsample is inherited from RelaxedOneHotCategorical'''
        u = torch.empty(self.logits.size(), device=self.logits.device, dtype=self.logits.dtype).uniform_(0, 1)
        noisy_logits = self.logits - torch.log(-torch.log(u))
        return torch.argmax(noisy_logits, dim=-1)

    def rsample(self, sample_shape=torch.Size()):
        '''
        Gumbel-softmax resampling using the Straight-Through trick.
        To see standalone code of how this works, refer to https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f
        '''
        rout = super().rsample(sample_shape)  # differentiable
        out = F.one_hot(torch.argmax(rout, dim=-1), self.logits.shape[-1]).float()
        return (out - rout).detach() + rout

        # Reparametrization trick.
        # return rout

    def log_prob(self, value):
        '''value is one-hot or relaxed'''
        if value.shape != self.logits.shape:
            value = F.one_hot(value.long(), self.logits.shape[-1]).float()
            assert value.shape == self.logits.shape
        return - torch.sum(- value * F.log_softmax(self.logits, -1), -1)
    
    def entropy(self):
        min_real = torch.finfo(self.logits.dtype).min
        logits = torch.clamp(self.logits, min=min_real)
        p_log_p = logits * self.probs
        return -p_log_p.sum(-1)


class ContinuousPolicy(Policy):
    """
    The policy is a Gaussian policy squashed by a tanh.
    """
    def __init__(self, mu_approximator, sigma_approximator,
                 min_a, max_a, log_std_min, log_std_max, temperature, gauss_noise_cov):
        """
        Constructor.
        Args:
            mu_approximator (Regressor): a regressor computing mean in a given
                state;
            sigma_approximator (Regressor): a regressor computing the variance
                in a given state;
            min_a (np.ndarray): a vector specifying the minimum action value
                for each component;
            max_a (np.ndarray): a vector specifying the maximum action value
                for each component.
            log_std_min ([float, Parameter]): min value for the policy log std;
            log_std_max ([float, Parameter]): max value for the policy log std;
            temperature ([float]): temperature for the Gumbel Softmax;
            gauss_noise_cov ([float]): Add gaussian noise to the drawn actions (if calling 'draw_noisy_action()')
        """
        self._mu_approximator = mu_approximator
        self._sigma_approximator = sigma_approximator
        
        self._temperature = torch.tensor(temperature)
        self._gauss_noise_cov = np.array(gauss_noise_cov)
        self._max_a = max_a
        self._min_a = min_a
        self._delta_a = to_float_tensor(.5 * (self._max_a - self._min_a), self.use_cuda)
        self._central_a = to_float_tensor(.5 * (self._max_a + self._min_a), self.use_cuda)

        self._log_std_min = to_parameter(log_std_min)
        self._log_std_max = to_parameter(log_std_max)

        self._eps_log_prob = 1e-6

        use_cuda = self._mu_approximator.model.use_cuda

        if use_cuda:
            self._delta_a = self._delta_a.cuda()
            self._central_a = self._central_a.cuda()

        self._add_save_attr(
            _mu_approximator='mushroom',
            _sigma_approximator='mushroom',
            _max_a='numpy',
            _min_a='numpy',
            _delta_a='torch',
            _central_a='torch',
            _log_std_min='mushroom',
            _log_std_max='mushroom',
            _eps_log_prob='primitive',
            _temperature='torch',
            _gauss_noise_cov='numpy'
        )

    def __call__(self, state, action):
        raise NotImplementedError

    def draw_action(self, state):
        return self.compute_action_and_log_prob_t(
            state, compute_log_prob=False).detach().cpu().numpy()

    def draw_deterministic_action(self, state):
        # Continuous        
        cont_mu_raw = self._mu_approximator.predict(state, output_tensor=True)
        a_cont = torch.tanh(cont_mu_raw)
        a_cont_true = a_cont * self._delta_a + self._central_a
        return a_cont_true.detach().cpu().numpy()

    def draw_noisy_action(self, state):
        # Add clipped gaussian noise (only to the continuous actions!)
        cont_noise = np.random.multivariate_normal(np.zeros(self._mu_approximator.output_shape[0]),np.eye(self._mu_approximator.output_shape[0])*self._gauss_noise_cov)
        noise = cont_noise
        return np.clip(self.compute_action_and_log_prob_t(state, compute_log_prob=False).detach().cpu().numpy() + noise, self._min_a, self._max_a)
    
    def compute_action_and_log_prob(self, state):
        """
        Function that samples actions using the reparametrization trick and
        the log probability for such actions.
        Args:
            state (np.ndarray): the state in which the action is sampled.
        Returns:
            The actions sampled and the log probability as numpy arrays.
        """
        a, log_prob = self.compute_action_and_log_prob_t(state)
        return a.detach().cpu().numpy(), log_prob.detach().cpu().numpy()

    def compute_action_and_log_prob_t(self, state, compute_log_prob=True):
        """
        Function that samples actions using the reparametrization trick and,
        optionally, the log probability for such actions.
        Args:
            state (np.ndarray): the state in which the action is sampled;
            compute_log_prob (bool, True): whether to compute the log
            probability or not.
        Returns:
            The actions sampled and, optionally, the log probability as torch
            tensors.
        """
        # Continuous
        cont_dist = self.cont_distribution(state)
        a_cont_raw = cont_dist.rsample()
        a_cont = torch.tanh(a_cont_raw)
        a_cont_true = a_cont * self._delta_a + self._central_a

        if compute_log_prob:
            # Continuous
            log_prob_cont = cont_dist.log_prob(a_cont_raw).sum(dim=1)
            log_prob_cont -= torch.log(1. - a_cont.pow(2) + self._eps_log_prob).sum(dim=1)
            
            return a_cont_true, log_prob_cont
        else:
            return a_cont_true

    def cont_distribution(self, state):
        """
        Compute the continous (Gaussian) policy distribution in the given states.
        Args:
            state (np.ndarray): the set of states where the distribution is
                computed.
        Returns:
            The torch distribution for the provided states.
        """
        mu = self._mu_approximator.predict(state, output_tensor=True)
        log_sigma = self._sigma_approximator.predict(state, output_tensor=True)
        # Bound the log_std
        log_sigma = torch.clamp(log_sigma, self._log_std_min(), self._log_std_max())
        return torch.distributions.Normal(mu, log_sigma.exp())


    def entropy(self, state=None):
        """
        Compute the entropy of the policy.
        Args:
            state (np.ndarray): the set of states to consider.
        Returns:
            The value of the entropy of the policy.
        """
        # Continuous dist and action
        cont_distr = self.cont_distribution(state)
        return torch.mean(cont_distr.entropy()).detach().cpu().numpy().item()

    def reset(self):
        pass

    def set_weights(self, weights):
        """
        Setter.
        Args:
            weights (np.ndarray): the vector of the new weights to be used by
                the policy.
        """
        mu_weights = weights[:self._mu_approximator.weights_size]
        sigma_weights = weights[self._mu_approximator.weights_size:self._mu_approximator.weights_size+self._sigma_approximator.weights_size]

        self._mu_approximator.set_weights(mu_weights)
        self._sigma_approximator.set_weights(sigma_weights)

    
    def get_weights(self):
        """
        Getter.
        Returns:
             The current policy weights.
        """
        mu_weights = self._mu_approximator.get_weights()
        sigma_weights = self._sigma_approximator.get_weights()

        return np.concatenate([mu_weights, sigma_weights])

    @property
    def use_cuda(self):
        """
        True if the policy is using cuda_tensors.
        """
        return self._mu_approximator.model.use_cuda

    def parameters(self):
        """
        Returns the trainable policy parameters, as expected by torch
        optimizers.
        Returns:
            List of parameters to be optimized.
        """
        return chain(self._mu_approximator.model.network.parameters(),
                     self._sigma_approximator.model.network.parameters())



class DiscretePolicy(Policy):
    """
    The policy is a Gaussian policy squashed by a tanh.
    """
    def __init__(self, discrete_approximator, min_a, max_a, log_std_min, log_std_max, temperature):
        """
        Constructor.
        Args:
            discrete_approximator (Regressor): a regressor computing the discrete
                action disctribution in a given state;
            min_a (np.ndarray): a vector specifying the minimum action value
                for each component;
            max_a (np.ndarray): a vector specifying the maximum action value
                for each component.
            log_std_min ([float, Parameter]): min value for the policy log std;
            log_std_max ([float, Parameter]): max value for the policy log std;
            temperature ([float]): temperature for the Gumbel Softmax;
        """
        self._discrete_approximator = discrete_approximator
        
        self._temperature = torch.tensor(temperature)
        self._max_a = max_a
        self._min_a = min_a

        self._log_std_min = to_parameter(log_std_min)
        self._log_std_max = to_parameter(log_std_max)

        self._eps_log_prob = 1e-6

        use_cuda = self._discrete_approximator.model.use_cuda


        self._add_save_attr(
            _discrete_approximator='mushroom',
            _max_a='numpy',
            _min_a='numpy',
            _log_std_min='mushroom',
            _log_std_max='mushroom',
            _eps_log_prob='primitive',
            _temperature='torch'
        )

    def __call__(self, state, action):
        raise NotImplementedError

    def draw_action(self, state):
        return self.compute_action_and_log_prob_t(
            state, compute_log_prob=False).detach().cpu().numpy()

    def draw_deterministic_action(self, state):        
        if isinstance(state, np.ndarray):
            if self._discrete_approximator.model.use_cuda:
                state = torch.from_numpy(state).cuda()
            else:
                state = torch.from_numpy(state)
        logits = None
        logits = self._discrete_approximator.predict(state, output_tensor=True)

        a_discrete = F.one_hot(torch.argmax(logits, dim=-1), logits.shape[-1])
        return a_discrete.detach().cpu().numpy()


    def compute_action_and_log_prob(self, state):
        """
        Function that samples actions using the reparametrization trick and
        the log probability for such actions.
        Args:
            state (np.ndarray): the state in which the action is sampled.
        Returns:
            The actions sampled and the log probability as numpy arrays.
        """
        a, log_prob = self.compute_action_and_log_prob_t(state)
        return a.detach().cpu().numpy(), log_prob.detach().cpu().numpy()

    def compute_action_and_log_prob_t(self, state, compute_log_prob=True):
        """
        Function that samples actions using the reparametrization trick and,
        optionally, the log probability for such actions.
        Args:
            state (np.ndarray): the state in which the action is sampled;
        Returns:
            The actions sampled and, optionally, the log probability as torch
            tensors.
        """
        # Discrete
        # NOTE: Discrete approximator takes both state and continuous action as input (sequential policy)
        
        discrete_dist = self.discrete_distribution(state)
        a_discrete = discrete_dist.rsample()

        if compute_log_prob:
            log_prob_discrete = discrete_dist.log_prob(a_discrete)
            return a_discrete, log_prob_discrete
        else:
            return a_discrete


    def discrete_distribution(self, state):
        """
        Compute the discrete policy distribution (categorical) in the given states.
        Args:
            state (np.ndarray): the set of states where the distribution is
                computed.
        Returns:
            The torch distribution for the provided states.
        """
        if isinstance(state, np.ndarray):
            if self._discrete_approximator.model.use_cuda:
                state = torch.from_numpy(state).cuda()
            else:
                state = torch.from_numpy(state)
        logits = None
        logits = self._discrete_approximator.predict(state, output_tensor=True)
        # print("Logits:", logits)
        return GumbelSoftmax(temperature=self._temperature, logits=logits)

    def entropy(self, state=None):
        """
        Compute the entropy of the policy.
        Args:
            state (np.ndarray): the set of states to consider.
        Returns:
            The value of the entropy of the policy.
        """
        return torch.mean(self.discrete_distribution(state).entropy()).detach().cpu().numpy().item()

    def reset(self):
        pass

    def get_weights(self):
        """
        Getter.
        Returns:
             The current policy weights.
        """
        discrete_weights = self._discrete_approximator.get_weights()

        return np.concatenate([discrete_weights])

    @property
    def use_cuda(self):
        """
        True if the policy is using cuda_tensors.
        """
        return self._discrete_approximator.model.use_cuda

    def parameters(self):
        """
        Returns the trainable policy parameters, as expected by torch
        optimizers.
        Returns:
            List of parameters to be optimized.
        """
        return chain(self._discrete_approximator.model.network.parameters())


class LRL(DeepAC):
    """
    LRL with a Hybrid action space
    """
    def __init__(self, mdp_info, action_space_range, actor_mu_params, actor_sigma_params, actor_discrete_params,
                 actor_optimizer, critic_params, batch_size,
                 initial_replay_size, max_replay_size, warmup_transitions, tau,
                 lr_alpha, log_std_min=-3, log_std_max=2, temperature=1.0, use_entropy=True, target_entropy=None,
                 gauss_noise_cov=0.01, critic_fit_params=None):
        """
        Constructor.
        Args:
            action_space_range (array): action space start and end for current policy
            actor_mu_params (dict): parameters of the actor mean approximator
                to build;
            actor_sigma_params (dict): parameters of the actor sigma
                approximator to build;
            actor_discrete_params (dict): parameters of the actor discrete distribution
                approximator to build;
            actor_optimizer (dict): parameters to specify the actor
                optimizer algorithm;
            critic_params (dict): parameters of the critic approximator to
                build;
            batch_size ((int, Parameter)): the number of samples in a batch;
            initial_replay_size (int): the number of samples to collect before
                starting the learning;
            max_replay_size (int): the maximum number of samples in the replay
                memory;
            warmup_transitions ([int, Parameter]): number of samples to accumulate in the
                replay memory to start the policy fitting;
            tau ([float, Parameter]): value of coefficient for soft updates;
            lr_alpha ([float, Parameter]): Learning rate for the entropy coefficient;
            log_std_min ([float, Parameter]): Min value for the policy log std;
            log_std_max ([float, Parameter]): Max value for the policy log std;
            temperature (float): the temperature for the softmax part of the gumbel reparametrization
            use_entropy (bool): Add entropy loss similar to SAC
            target_entropy (float, None): target entropy for the policy, if
                None a default value is computed ;
            gauss_noise_cov ([float, Parameter]): Add gaussian noise to the drawn actions (if calling 'draw_noisy_action()');
            critic_fit_params (dict, None): parameters of the fitting algorithm
                of the critic approximator.
        """
        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params

        self._batch_size = to_parameter(batch_size)
        self._warmup_transitions = to_parameter(warmup_transitions)
        self._tau = to_parameter(tau)
        self._action_space_range = np.array(action_space_range)
        self._use_prior = False

        if target_entropy is None:
            self._target_entropy = -np.prod(mdp_info.action_space.shape).astype(np.float32)            
        else:
            self._target_entropy = target_entropy

        self._replay_memory = ReplayMemory(initial_replay_size, max_replay_size)

        if 'n_models' in critic_params.keys():
            assert critic_params['n_models'] == 2
        else:
            critic_params['n_models'] = 2

        target_critic_params = deepcopy(critic_params)

        self._critic_approximator = Regressor(TorchApproximator,
                                            **critic_params)
        self._target_critic_approximator = Regressor(TorchApproximator,
                                                    **target_critic_params)
        

        # self._state_dim = actor_mu_params['input_shape'][0] 

        self._use_entropy = use_entropy

        actor_mu_approximator = None
        actor_sigma_approximator = None
        actor_discrete_approximator = None
        policy = None
        policy_parameters = None

        if actor_mu_params:
            actor_mu_approximator = Regressor(TorchApproximator,
                                            **actor_mu_params)
            actor_sigma_approximator = Regressor(TorchApproximator,
                                                **actor_sigma_params)


            policy = ContinuousPolicy(actor_mu_approximator,
                        actor_sigma_approximator,
                        mdp_info.action_space.low,
                        mdp_info.action_space.high,
                        log_std_min,
                        log_std_max,
                        temperature,
                        gauss_noise_cov)

            policy_parameters = chain(actor_mu_approximator.model.network.parameters(),
                                actor_sigma_approximator.model.network.parameters())

        if actor_discrete_params:
            actor_discrete_approximator = Regressor(TorchApproximator,
                                                **actor_discrete_params)

            policy = DiscretePolicy(actor_discrete_approximator,
                        mdp_info.action_space.low[action_space_range[0]:action_space_range[1]],
                        mdp_info.action_space.high[action_space_range[0]:action_space_range[1]],
                        log_std_min,
                        log_std_max,
                        temperature)

            policy_parameters = chain(actor_discrete_approximator.model.network.parameters())

        self._actor_last_loss = None # Store actor loss for logging
        
        self._init_target(self._critic_approximator,
                          self._target_critic_approximator)

        self._log_alpha = torch.tensor(0., dtype=torch.float32)

        if policy.use_cuda:
            self._log_alpha = self._log_alpha.cuda().requires_grad_()
        else:
            self._log_alpha.requires_grad_()

        self._alpha_optim = optim.Adam([self._log_alpha], lr=lr_alpha)

        

        self._add_save_attr(
            _critic_fit_params='pickle',
            _batch_size='mushroom',
            _warmup_transitions='mushroom',
            _tau='mushroom',
            _target_entropy='primitive',
            _replay_memory='mushroom',
            _critic_approximator='mushroom',
            _target_critic_approximator='mushroom',
            _state_dim='primitive',
            _use_entropy='primitive',
            _log_alpha='torch',
            _alpha_optim='torch',
            _action_space_range='primitive'
        )

        super().__init__(mdp_info, policy, actor_optimizer, policy_parameters)

    def setup_prior(self, prior_agents, state_dims, action_dims, action_space_range=None):
        """
            prior_agents ([mushroom object list]): The agent object from agents trained on prior tasks (order is impt);
            state_dims (np.ndarray): state dims of prior agents
            action_dims (np.ndarray): action dims of prior agents
        """
        self._use_prior = True
        self._prior_critic_approximators = list()
        self._prior_policies = list()
        self._prior_state_dims = np.array(state_dims)
        self._prior_action_dims = np.array(action_dims)
        self._prior_agents = list()

        if action_space_range:
            self._action_space_range = np.array(action_space_range)

        for prior_agent in prior_agents:
            # self._prior_critic_approximators.append(deepcopy(prior_agent._target_critic_approximator)) # The target_critic_approximator object from agents trained on prior tasks
            self._prior_critic_approximators.append(prior_agent._target_critic_approximator) # The target_critic_approximator object from agents trained on prior tasks
            self._prior_policies.append(prior_agent.policy) # The policy object from an agent trained on a prior task

            # for learning
            # prior_agent._target_critic_approximator.model.reset()
            self._prior_agents.append(prior_agent)

    # Overidden from agent.py
    def draw_action(self, state):
        """
        Return the action to execute in the given state. It is the action
        returned by the policy or the action set by the algorithm.

        Args:
            state (np.ndarray): the state where the agent is.

        Returns:
            The action to be executed.

        """
        if self.phi is not None:
            state = self.phi(state)

        if self.next_action is None:
            if self._use_prior:
                n_prior_agent_actions = np.sum(self._prior_action_dims[:,1] - self._prior_action_dims[:,0])
                n_curr_agent_actions = self._action_space_range[1] - self._action_space_range[0]
                actions = np.zeros((n_prior_agent_actions+n_curr_agent_actions,))

                for i, prior_policy in enumerate(self._prior_policies):
                    if i == 0:
                        state_t = state[self._prior_state_dims[i,0]:self._prior_state_dims[i,1]]
                    else:
                        state_t = np.hstack((state[self._prior_state_dims[i,0]:self._prior_state_dims[i,1]], \
                            actions[self._prior_action_dims[0,0]:self._prior_action_dims[i-1,1]]))
                    
                    actions[self._prior_action_dims[i,0]:self._prior_action_dims[i,1]] = self._prior_policies[i].draw_action(state_t)

                state = np.hstack((state, actions[self._prior_action_dims[0,0]:self._prior_action_dims[i,1]]))
                actions[self._action_space_range[0]:self._action_space_range[1]] = self.policy.draw_action(state)
                return actions
            else:
                n_curr_agent_actions = self._action_space_range[1] - self._action_space_range[0]
                actions = np.zeros((n_curr_agent_actions,))
                actions[self._action_space_range[0]:self._action_space_range[1]] = self.policy.draw_action(state)
                return actions
        else:
            action = self.next_action
            self.next_action = None
            return action

    def _get_prior_action(self, state):
        n_prior_agent_actions = np.sum(self._prior_action_dims[:,1] - self._prior_action_dims[:,0])
        actions = np.zeros((len(state), n_prior_agent_actions))

        for i, prior_policy in enumerate(self._prior_policies):
            if i == 0:
                state_t = state[:,self._prior_state_dims[i,0]:self._prior_state_dims[i,1]]
            else:
                state_t = np.hstack((state[:,self._prior_state_dims[i,0]:self._prior_state_dims[i,1]], \
                    actions[:,self._prior_action_dims[0,0]:self._prior_action_dims[i-1,1]]))
            actions[:,self._prior_action_dims[i,0]:self._prior_action_dims[i,1]] = self._prior_policies[i].draw_action(state_t)
        return actions        

    def fit(self, dataset):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _ = \
                self._replay_memory.get(self._batch_size())

            next_action_prior = self._get_prior_action(next_state)

            q_next = self._next_q(np.hstack((next_state,next_action_prior)), absorbing)
            q = reward + self.mdp_info.gamma * q_next
            rho = q

            rho_optimal = 0
            for i, prior_critic in enumerate(self._prior_critic_approximators):
                t_state = None                
                if i == 0:
                    t_state = state[:,self._prior_state_dims[i][0]:self._prior_state_dims[i][1]]
                else:
                    t_state = np.hstack((state[:,self._prior_state_dims[i][0]:self._prior_state_dims[i][1]], \
                        action[:,self._prior_action_dims[0][0]:self._prior_action_dims[i-1][1]]))

                t_action = action[:,self._prior_action_dims[i][0]:self._prior_action_dims[i][1]]
                
                rho_optimal = rho_optimal + prior_critic.predict(t_state, t_action, prediction='min', output_tensor=True).values



            if self._replay_memory.size > self._warmup_transitions():

                prior_action_start = self._prior_action_dims[0][0]
                prior_action_end = self._prior_action_dims[-1][1]

                action_new, log_prob = self.policy.compute_action_and_log_prob_t(np.hstack((state,
                    action[:,prior_action_start:prior_action_end])))

                loss = self._loss(np.hstack((state,action[:,prior_action_start:prior_action_end])), 
                    action_new, log_prob, rho_optimal, []) # add rho priors here for growing action spaces
                self._optimize_actor_parameters(loss)

                if self._use_entropy:
                    self._update_alpha(log_prob.detach())
                self._actor_last_loss = loss.detach().cpu().numpy() # Store actor loss for logging

            

            
            self._critic_approximator.fit(state, action, rho,
                                          **self._critic_fit_params)

            self._update_target(self._critic_approximator,
                                self._target_critic_approximator)

    
    def _loss_prior(self, state, action_new, log_prob):
        rho_0 = self.prior_agents[0]._critic_approximator(state, action_new,
                                        output_tensor=True, idx=0)
        rho_1 = self.prior_agents[0]._critic_approximator(state, action_new,
                                        output_tensor=True, idx=1)

        q = torch.min(rho_0, rho_1)
            
        if self._use_entropy:
            q -= self._alpha * log_prob

        return  -q.mean()

    def _loss(self, state, action_new, log_prob, rho_optimal, rho_priors):
        rho_0 = self._critic_approximator(state, action_new,
                                        output_tensor=True, idx=0)
        rho_1 = self._critic_approximator(state, action_new,
                                        output_tensor=True, idx=1)

        q = torch.min(rho_0, rho_1)

        if self._use_entropy:
            q -= self._alpha * log_prob

        return -q.mean()

    def _update_alpha(self, log_prob):
        alpha_loss = - (self._log_alpha * (log_prob + self._target_entropy)).mean()
        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_optim.step()

    # TODO generic for all priors
    def _next_q_prior(self, next_state, absorbing):
        """
        Args:
            next_state (np.ndarray): the states where next action has to be
                evaluated;
            absorbing (np.ndarray): the absorbing flag for the states in
                ``next_state``.
        Returns:
            Action-values returned by the critic for ``next_state`` and the
            action returned by the actor.
        """

        a, log_prob_next = self._prior_agents[0].policy.compute_action_and_log_prob(next_state)

        q = self._prior_agents[0]._target_critic_approximator.predict(
            next_state, a, prediction='min')
        
        if self._use_entropy:
            q -= self._alpha_np * log_prob_next

        q *= 1 - absorbing
        # print("Q absorbing:", q, absorbing)
        return q

    def _next_q(self, next_state, absorbing):
        """
        Args:
            next_state (np.ndarray): the states where next action has to be
                evaluated;
            absorbing (np.ndarray): the absorbing flag for the states in
                ``next_state``.
        Returns:
            Action-values returned by the critic for ``next_state`` and the
            action returned by the actor.
        """
        a, log_prob_next = self.policy.compute_action_and_log_prob(next_state)

        q = self._target_critic_approximator.predict(
            next_state, a, prediction='min')
        
        if self._use_entropy:
            q -= self._alpha_np * log_prob_next

        q *= 1 - absorbing

        return q

    def _post_load(self):
        self._update_optimizer_parameters(self.policy.parameters())

    @property
    def _alpha(self):
        return self._log_alpha.exp()

    @property
    def _alpha_np(self):
        return self._alpha.detach().cpu().numpy()