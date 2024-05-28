from typing import Dict

import numpy as np
import torch
from utils import TorchStandardScaler


class ReplayBuffer:

    def __init__(self,
        state_dim: int,
        action_dim: int,
        max_size: int = int(1e6),
        device: str = "cpu"):
        """
        Constructor.

        :param state_dim: State dimension.
        :param action_dim: Action dimension.
        :parma max_size: Size of the replay buffer.
        :param device: Device to store the tensor when sampling the replay buffer.
        :param scaler_file_name: scaler (normalizer) file name for loading
        :param seed: seed for loading a replay buffer of a random seed
        """
        self.device = torch.device(device)
        self.max_size = max_size
        
        ## property of the transition buffer
        self.ptr = 0
        self.size = 0

        ## property of the starting state buffer
        self.start_ptr = 0
        self.start_size = 0

        ## main content of the buffer
        self.state = torch.zeros((max_size, state_dim), device=self.device)
        self.action = torch.zeros((max_size, action_dim), device=self.device)
        self.next_state = torch.zeros((max_size, state_dim), device=self.device)
        self.next_action = torch.zeros((max_size, action_dim), device=self.device)
        self.reward = torch.zeros((max_size, 1), device=self.device)
        self.not_done = torch.zeros((max_size, 1), device=self.device)
        self.start_state = torch.zeros((max_size, state_dim), device=self.device)
        self.state_scaler = None

        

    def add(self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        next_action: np.ndarray,
        reward: float,
        done: float) -> None:
        """
        Add transition to the transition buffer.

        :param state: Current state array of shape (state_dim).
        :param action: Current action array of shape (action_dim).
        :param next_state: Next state array of shape (state_dim).
        :param reward: Reward.
        :param done: Done signal.
        """
        self.state[self.ptr] = torch.tensor(state, device=self.device)
        self.action[self.ptr] = torch.tensor(action, device=self.device)
        self.next_state[self.ptr] = torch.tensor(next_state, device=self.device)
        self.next_action[self.ptr] = torch.tensor(next_action, device=self.device)
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def add_start(self, state: np.ndarray) -> None:
        """
        Add starting state to the starting state buffer.
        :param state: Starting state array of shape (state_dim).
        """
        self.start_state[self.start_ptr] = torch.tensor(state, device=self.device)

        self.start_ptr = (self.start_ptr + 1) % self.max_size
        self.start_size = min(self.start_size + 1, self.max_size)


    def sample_start_state(self, batch_size: int = -1, scaled:bool = False) -> torch.Tensor:
        """
        Sample starting states from the starting state buffer.

        :param batch_size: Batch size. -1 means get all datapoints.
        :param scaled: True to get normalized data, False for unnormalized original data

        Return:
            Tensor of starting states of shape (batch_size, state_dim).
        """
        if batch_size == -1:
            idx = np.arange(self.start_size)
        else:
            idx = np.random.randint(self.start_size, size=batch_size)

        if not scaled:
            return self.start_state[idx]
        else:
            assert self.state_scaler != None
            scaled_start_state =  self.state_scaler.transform(self.start_state[idx])
            return scaled_start_state

    def sample(self, batch_size: int, scaled:bool = False, resample_idx:torch.tensor =None) -> Dict[str, torch.Tensor]:
        """
        Sample transitions from the transition buffer.

        :param batch_size: Batch size.
        :param scaled: True to get normalized data (but the actions are not scaled)
        :param resample_idx: get the values by idx sampled with resampling

        Return:
            Dictionary with the following keys and values:
            - "state": Tensor of shape (batch_size, state_dim)
            - "action": Tensor of shape (batch_size, action_dim)
            - "next_state": Tensor of shape (batch_size, next_state)
            - "reward": Tensor of shape (batch_size, 1)
            - "not_done": Tensor of shape (batch_size, 1)
        """

        if resample_idx is None:
            idx = np.random.randint(self.size, size=batch_size)
        else:
            idx = resample_idx

        if not scaled:
            return {
                "state":  self.state[idx],
                "action": self.action[idx],
                "next_state": self.next_state[idx],
                "next_action": self.next_action[idx],
                "reward":   self.reward[idx],
                "not_done": self.not_done[idx]
            }

        else:
            assert self.state_scaler != None
            scaled_state      = self.state_scaler.transform(self.state[idx])
            scaled_next_state = self.state_scaler.transform(self.next_state[idx])

            return {
                "state":      scaled_state,
                "action":     self.action[idx],
                "next_state": scaled_next_state,
                "next_action": self.next_action[idx],
                "reward":     self.reward[idx],
                "not_done":   self.not_done[idx]
            }


    def fit_scalers(self,):
        """
        fit scalers for states (next states) and rewards to normalize them
        execute this function after collecting data
        """
        self.state_scaler  = TorchStandardScaler()
        self.state_scaler.fit( self.state[:self.size])


    def get_scalers(self,):
        self.state_scaler.to_device(self.device)
        return self.state_scaler

    def set_device(self, device_name:str):
        # set the device attribute to the designated one.
        self.device = torch.device(device_name)

        if self.state_scaler!=None:
            self.state_scaler.to_device(self.device)


    def get_all_data(self, scaled:bool) -> Dict[str, torch.Tensor]:
        if not scaled:
            return {
                "state":  self.state[:self.size],
                "action": self.action[:self.size],
                "next_state": self.next_state[:self.size],
                "next_action": self.next_action[:self.size],
                "reward":   self.reward[:self.size],
                "not_done": self.not_done[:self.size]
            }

        else:
            assert self.state_scaler != None
            scaled_state      = self.state_scaler.transform(self.state[:self.size])
            scaled_next_state = self.state_scaler.transform(self.next_state[:self.size])

            return {
                "state": scaled_state,
                "action": self.action[:self.size],
                "next_state": scaled_next_state,
                "next_action": self.next_action[:self.size],
                "reward":   self.reward[:self.size],
                "not_done": self.not_done[:self.size]
            }
