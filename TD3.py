from typing import Tuple

import copy
import numpy as np
import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F

from replay_buffer import ReplayBuffer


class Actor(nn.Module):
    """
    TD3 Deterministic Actor.
    """

    def __init__(self, state_dim: int, action_dim: int, max_action: int, n_dummy_dim: int = 0):
        """
        Constructor.

        :param state_dim: State dimension.
        :param action_dim: Action dimension.
        :param max_action: Maximum value of action.
        """
        super().__init__()

        self.n_dummy_dim = n_dummy_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        :param state: Tensor of state of shape (batch_size, state_dim).

        Return:
            Tensor of action of shape (batch_size, action_dim).
        """

        if self.n_dummy_dim == 0:
            action = self.max_action * self.net.forward(state)
            return action
        else:
            ori_action   = self.max_action * self.net.forward(state) # original action

            device_info = ori_action.get_device()
            device_str = ''
            if device_info==-1:
                device_str = 'cpu'
            else:
                device_str = f'cuda:{device_info:d}'

            dummy_action = torch.zeros([ori_action.shape[0],self.n_dummy_dim], device=device_str)
            action = torch.hstack((ori_action, dummy_action))

            return action


class Critic(nn.Module):
    """
    TD3 Critic.
    """

    def __init__(self, state_dim: int, action_dim: int):
        """
        Constructor.

        :param state_dim: State dimension.
        :param action_dim: Action dimension.
        """
        super().__init__()

        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        :param state: Tensor of state of shape (batch_size, state_dim).
        :param action: Tensor of action of shape (batch_size, action_dim).

        Return:
            Tuple of q-values with both shape of (batch_size, 1).
        """
        x = torch.cat([state, action], dim=1)
        q1_val = self.q1_net.forward(x)
        q2_val = self.q2_net.forward(x)
        return (q1_val, q2_val)

    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get Q1 value only.

        :param state: Tensor of state of shape (batch_size, state_dim).
        :param action: Tensor of action of shape (batch_size, action_dim).

        Return:
            Tensor of Q1 values of shape (batch_size, 1).
        """
        x = torch.cat([state, action], dim=1)
        return self.q1_net.forward(x)


class TD3:
    """
    Twin-Delayed DDPG.
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            max_action: int,
            discount: float = 0.99,
            tau: float = 0.005,
            policy_noise: float = 0.2,
            noise_clip: float = 0.5,
            policy_freq: int = 2,
            n_dummy_dim: int = 0,
            device: str = "cuda:0"):
        """
        Constructor.

        :param state_dim: State dimension.
        :param action_dim: Action dimension.
        :param max_action: Maximum action.
        :param discount: Dicount factor.
        :param tau: Polyak averaging constant for target network update.
        :param policy_noise: Std for policy during training.
        :param noise_clip: Highest values of noise allowed.
        :param policy_freq: How often to update the policy (delayed update).
        :param device: Device.
        """
        self.device = torch.device(device)

        self.actor = Actor(state_dim, action_dim, max_action, n_dummy_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.n_dummy_dim = n_dummy_dim

        self.total_it = 0

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Get action from the actor.

        :param state: Numpy array of state of shape (state_dim).

        Return:
            Numpy array of action of shape (action_dim).
        """
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = self.actor.forward(state).detach().cpu().numpy()
        return action

    def train(self, replay_buffer: ReplayBuffer, batch_size: int) -> None:
        """
        One step of gradient update using TD3 objective.

        :param replay_buffer. Replay buffer containing the training data.
        :param batch_size: Batch size for one step update.
        """
        self.total_it += 1

        ## sample replay buffer
        batch = replay_buffer.sample(batch_size, scaled=False)
        state = batch["state"]
        action = batch["action"]
        next_state = batch["next_state"]
        reward = batch["reward"]
        not_done = batch["not_done"]

        with torch.no_grad():
            ## select action according to policy and add noise
            noise = torch.randn_like(action) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)

            next_action = self.actor_target.forward(next_state) + noise
            next_action = next_action.clamp(-self.max_action, self.max_action)

            ## compute the target Q values
            target_q1, target_q2 = self.critic_target.forward(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.discount * target_q

        ## get current Q values
        current_q1, current_q2 = self.critic.forward(state, action)

        ## compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        ## optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        ## delayed policy updates
        if self.total_it % self.policy_freq == 0:
            ## compute actor loss
            actor_loss = - self.critic.q1(state, self.actor.forward(state)).mean()

            ## optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            ## Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename: str) -> None:
        """
        Save the model.

        :param filename: Name of the directory.
        """
        torch.save(self.critic.state_dict(), os.path.join(filename, "critic"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(filename, "critic_optimizer"))

        torch.save(self.actor.state_dict(), os.path.join(filename, "actor"))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(filename, "actor_optimizer"))

    def load(self, filename: str) -> None:
        """
        Load the model.

        :param filename: Name of the directory.
        """
        self.critic.load_state_dict(torch.load(os.path.join(filename, "critic"), map_location=self.device))
        self.critic_optimizer.load_state_dict(
            torch.load(os.path.join(filename, "critic_optimizer"), map_location=self.device))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(os.path.join(filename, "actor"), map_location=self.device))
        self.actor_optimizer.load_state_dict(
            torch.load(os.path.join(filename, "actor_optimizer"), map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor)
