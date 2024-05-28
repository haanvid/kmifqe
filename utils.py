import pandas as pd
import os
import torch
from functorch import make_functional_with_buffers, vmap, grad, hessian


def get_critic_hess_action(model, state, action):
    # returns a tuple (len n_layer*2) (2 for weight and bias) containing grads w.r.t. params
    # model: torch.nn.Module
    # state, action: shape: (bs, state_dim), (bs, act_dim)
    # targets: true values for computing losses

    fmodel, params, buffers = make_functional_with_buffers(model)

    def compute_prediction_stateless_model(params, buffers, state, action):
        state = state.unsqueeze(0)
        action = action.unsqueeze(0)

        predictions = fmodel(params, buffers, state, action)
        return torch.squeeze(predictions)

    ft_compute_hess = hessian(compute_prediction_stateless_model, argnums=3)  # gets grad w.r.t. action
    ft_compute_batch_hess = vmap(ft_compute_hess, in_dims=(None, None, 0, 0))
    ft_batch_hesses = ft_compute_batch_hess(params, buffers, state, action)

    return ft_batch_hesses

def get_critic_grad_param(model, state, action):
    # returns a tuple (len n_layer*2) (2 for weight and bias) containing grads w.r.t. params
    # model: torch.nn.Module
    # state, action: shape: (bs, state_dim), (bs, act_dim)
    # targets: true values for computing losses

    fmodel, params, buffers = make_functional_with_buffers(model)

    def compute_prediction_stateless_model(params, buffers, state, action):
        state = state.unsqueeze(0)
        action = action.unsqueeze(0)

        predictions = fmodel(params, buffers, state, action)
        return torch.squeeze(predictions)

    ft_compute_grad = grad(compute_prediction_stateless_model, argnums=0)  # gets grad w.r.t. params
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
    ft_per_sample_grads = ft_compute_sample_grad(params, buffers, state, action)

    return ft_per_sample_grads


class TorchStandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
    def fit(self, x):
        self.mean = x.mean(0, keepdim=False)
        self.std = x.std(0, unbiased=False, keepdim=False)
    def transform(self, x, EPS=1e-8):
        # assert all(i >= 0.0 for i in self.std)
        x_scaled = x - self.mean
        x_scaled = x_scaled / (self.std + EPS)
        return x_scaled
    def inverse_transform(self, x):
        x_inv_scaled = x * self.std
        x_inv_scaled = x_inv_scaled + self.mean
        return x_inv_scaled

    def to_device(self, device):
        # device is torch.device
        self.mean = self.mean.to(device)
        self.std  = self.std.to(device)


def make_dir_with_dict(dict, start_dir):
    # loop through the dictionary keys to create subdirectories and files
    path = start_dir
    for dict_key in dict.keys():
        path = os.path.join(path, f"{dict_key}_{dict[dict_key]}")
    os.makedirs(path, exist_ok=True)

    return path


def write_csv(i_train_step, squared_err, eval, gt_policy_val, csv_path):

    data = {
        'i_train_step': [i_train_step],
        'squared_err': [squared_err],
        'eval': [eval],
        'gt_policy_val': [gt_policy_val]
    }

    # Make data frame of above data
    df = pd.DataFrame(data)

    # append data frame to CSV file
    if i_train_step==0:
        df.to_csv(csv_path, mode='w', index=False, header=True)
    else:
        df.to_csv(csv_path, mode='a', index=False, header=False)

    return

