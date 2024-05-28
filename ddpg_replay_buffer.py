from typing import Dict

import numpy as np
import torch
from utils import TorchStandardScaler


class ReplayBuffer:
    """
    Replay buffer.
    scaled=True : normalize the data contained in the replay buffer except the actions as they are generally bounded
    """

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
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((max_size, 1), dtype=np.float32)
        self.start_state = np.zeros((max_size, state_dim), dtype=np.float32)

        self.state_scaler = None



    def add(self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float,
        done: bool) -> None:
        """
        Add transition to the transition buffer.
        :param state: Current state array of shape (state_dim).
        :param action: Current action array of shape (action_dim).
        :param next_state: Next state array of shape (state_dim).
        :param reward: Reward.
        :param done: Done signal.
        """
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def add_start(self, state: np.ndarray) -> None:
        """
        Add starting state to the starting state buffer.
        :param state: Starting state array of shape (state_dim).
        """
        self.start_state[self.start_ptr] = state

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
            return torch.tensor(self.start_state[idx], device=self.device)
        else:
            assert self.state_scaler != None
            scaled_start_state =  self.state_scaler.transform(torch.tensor(self.start_state[idx], device=self.device))
            return scaled_start_state

    def sample(self, batch_size: int, scaled:bool = False) -> Dict[str, torch.Tensor]:
        """
        Sample transitions from the transition buffer.
        :param batch_size: Batch size.
        :param scaled: True to get normalized data (but the actions are not scaled)
        Return:
            Dictionary with the following keys and values:
            - "state": Tensor of shape (batch_size, state_dim)
            - "action": Tensor of shape (batch_size, action_dim)
            - "next_state": Tensor of shape (batch_size, next_state)
            - "reward": Tensor of shape (batch_size, 1)
            - "not_done": Tensor of shape (batch_size, 1)
        """
        idx = np.random.randint(self.size, size=batch_size)

        if not scaled:
            return {
                "state":  torch.tensor(self.state[idx], device=self.device),
                "action": torch.tensor(self.action[idx], device=self.device),
                "next_state": torch.tensor(self.next_state[idx], device=self.device),
                "reward":   torch.tensor(self.reward[idx], device=self.device),
                "not_done": torch.tensor(self.not_done[idx], device=self.device)
            }

        else:
            assert self.state_scaler != None
            scaled_state      = self.state_scaler.transform(torch.tensor(self.state[idx], device=self.device))
            scaled_next_state = self.state_scaler.transform(torch.tensor(self.next_state[idx], device=self.device))

            return {
                "state":      scaled_state,
                "action":     torch.tensor(self.action[idx], device=self.device),
                "next_state": scaled_next_state,
                "reward":     torch.tensor(self.reward[idx], device=self.device),
                "not_done":   torch.tensor(self.not_done[idx], device=self.device)
            }


    def fit_scalers(self,):
        """
        fit scalers for states (next states) and rewards to normalize them
        execute this function after collecting data
        """
        # Use TorchStandardScalers
        self.state_scaler  = TorchStandardScaler()
        self.state_scaler.fit( torch.tensor(self.state[:self.size], device=self.device))


    def get_scalers(self,):
        self.state_scaler.to_device(self.device)
        return self.state_scaler

    def set_device(self, device_name:str):
        # set the device attribute to the designated one.
        self.device = torch.device(device_name)

        # make the means and stds in scalers to be in same device as the tensors and return the scalers
        if self.state_scaler!=None:
            self.state_scaler.to_device(self.device)


    def get_all_data(self, scaled:bool) -> Dict[str, torch.Tensor]:
        if not scaled:
            return {
                "state":  torch.tensor(self.state[:self.size], device=self.device),
                "action": torch.tensor(self.action[:self.size], device=self.device),
                "next_state": torch.tensor(self.next_state[:self.size], device=self.device),
                "reward":   torch.tensor(self.reward[:self.size], device=self.device),
                "not_done": torch.tensor(self.not_done[:self.size], device=self.device)
            }

        else:
            assert self.state_scaler != None
            scaled_state      = self.state_scaler.transform(torch.tensor(self.state[:self.size], device=self.device))
            scaled_next_state = self.state_scaler.transform(torch.tensor(self.next_state[:self.size], device=self.device))

            return {
                "state": scaled_state,
                "action": torch.tensor(self.action[:self.size], device=self.device),
                "next_state": scaled_next_state,
                "reward":   torch.tensor(self.reward[:self.size], device=self.device),
                "not_done": torch.tensor(self.not_done[:self.size], device=self.device)
            }