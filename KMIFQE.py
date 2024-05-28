import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import functorch
import wandb
from functorch import hessian
from utils import get_critic_hess_action, get_critic_grad_param


def gaussian_kernel(u):

	assert len(u.shape)==1
	with torch.no_grad():
		kernel_val = torch.exp(-0.5 * torch.inner(u, u)) / ((2.0 * torch.pi) ** (len(u) / 2.0))
	return kernel_val

def gaussian_kernel_dim_wise(u):
	assert len(u.shape)==1
	gaussian_kernel_dim_wise = torch.exp(-0.5 * u**2.0 ) / ((2.0 * torch.pi) ** (0.5))

	return gaussian_kernel_dim_wise


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_units=256, last_layer_activation=None):
		super(Critic, self).__init__()
		self.l1 = nn.Linear(state_dim + action_dim, hidden_units)
		self.l2 = nn.Linear(hidden_units, hidden_units)
		self.l3 = nn.Linear(hidden_units, 1)
		self.last_layer_activation=last_layer_activation

	def forward(self, state, action):
		q1 = torch.tanh(self.l1(torch.cat([state, action], 1)))
		q1 = torch.tanh(self.l2(q1))
		# return self.l3(q1)
		if self.last_layer_activation == "exp":
			return torch.exp(self.l3(q1))
		else:
			return self.l3(q1)


class KMIFQE(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		last_layer_activation=None,
		hidden_units = 256,
		discount=0.99,
		tau=0.005,
		clip_behav_den_val_min=0.0,
		dim_wise_is_clip=False,
		critic_target_path = "./critic_targets/critic_target",
		normalized_action_value=False,
		relax_target_std=0.0,
		random=0.0,
		behav_bias=0.0,
		behav_std=0.2,
		clip_val_max=1e8,
		clip_val_min=1e-8,
		batch_size= 256,
		hessian_batch_size = 256,
		h_batch_size = 256,
		reg_multiplier = 0.1,
		weight_decay = 0.0,
		max_episode_len = None,
		n_eval_freq = 1000,
		env_is_ant = False,
		device='cpu'
	):
		self.device = torch.device(device)
		self.last_layer_activation = last_layer_activation
		self.critic = Critic(state_dim, action_dim, hidden_units, last_layer_activation).to(self.device)
		self.critic_target = copy.deepcopy(self.critic)
		self.weight_decay = weight_decay
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4, weight_decay=self.weight_decay)

		self.discount = discount
		self.tau = tau
		self.clip_behav_den_val_min = clip_behav_den_val_min
		self.dim_wise_is_clip = dim_wise_is_clip
		self.critic_target_path = critic_target_path
		self.normalized_action_value = normalized_action_value

		self.total_it = 0
		self.hidden_units = hidden_units
		self.action_dim = action_dim
		self.max_action = max_action
		self.relax_target_std = relax_target_std
		self.random = random
		self.behav_bias = behav_bias
		self.behav_std  = behav_std
		self.clip_val_max = clip_val_max
		self.clip_val_min = clip_val_min
		self.batch_size = batch_size
		self.hessian_batch_size = hessian_batch_size
		self.h_batch_size = h_batch_size
		self.reg_multiplier = reg_multiplier
		self.weight_decay = weight_decay
		self.max_episode_len = max_episode_len
		self.n_eval_freq = n_eval_freq
		self.env_is_ant = env_is_ant


	def get_behav_pol_den(self, action, mean):
		# Gaussian pdf value
		# action, target action: not standardized
		# random: probability of action sampled from the uniform distribution
		# behav_std: behavior action std of each action dimension
		# returns mixture of Gaussian (which mean is at "target_action") + uniform density value at "action"
		with torch.no_grad():
			assert len(action.shape) == 2
			batch_size = action.shape[0]
			behav_pol_std = self.max_action * self.behav_std * torch.ones(batch_size, device=self.device)
			behav_act_dist = Normal(loc=mean[:, 0], scale=behav_pol_std)
			behav_act_log_prob = behav_act_dist.log_prob(action[:, 0])
			behav_pol_den = torch.exp(behav_act_log_prob)
			behav_pol_den = (1.0 - self.random) * behav_pol_den + self.random * (2.0 * self.max_action) ** (-1.0)
			behav_pol_den = behav_pol_den * (2.0 * self.max_action) ** (-float(self.action_dim - 1.0))

		return behav_pol_den # shape: (batch_size,)


	def get_gauss_behav_pol_den_dim_wise(self, action, mean):
		# returns Gaussian pdf value of each dim
		# action, behav_det_action: not standardized
		# behav_det_action: deterministic action used to make Gaussian behavior policy
		# behav_std: behavior action std of each action dimension

		assert len(action.shape) == 2
		batch_size = action.shape[0]
		behav_pol_std = self.max_action * self.behav_std * torch.ones(batch_size, device=self.device)
		behav_act_dist = Normal(loc=mean[:, 0], scale=behav_pol_std)
		behav_act_log_prob = behav_act_dist.log_prob(action[:, 0])
		behav_pol_den_1st_dim = torch.exp(behav_act_log_prob)
		gauss_behav_pol_den = torch.ones(batch_size, self.action_dim, device=self.device) * (2.0 * self.max_action) ** (-1.0)
		gauss_behav_pol_den[:, 0] = behav_pol_den_1st_dim

		return gauss_behav_pol_den # shape: (batch_size, act_dim)


	def get_h(self, tr_B, state, action, reward, next_state, next_target_action, behav_pol_den_target, i_train_step, wandb_result=None):

		weight_grads = get_critic_grad_param(self.critic, state, action)

		with torch.no_grad(): # without this memory explodes due to accumulation of backward graphs
			grad_flattened_list = []
			for weight_grad in weight_grads:
				weight_grad_flattened = torch.flatten(weight_grad, start_dim=1) # (batch_size, layer_param)
				grad_flattened_list.append(weight_grad_flattened)

			grad_flattened = torch.hstack(grad_flattened_list) # (batch_size, n_param)

			# for bias
			tr_B = torch.unsqueeze(tr_B, dim=1)  # (batch_size, 1) for broadcasting
			X_b = torch.sum(tr_B * grad_flattened, dim=0) # shape (bs, 1 )*(bs, n_param )

			# for variance
			grad_l2_squared = torch.sum(grad_flattened ** 2.0, dim = 1) # (batch_size, )
			if self.normalized_action_value:
				td_err_squared = ((1.0-self.discount) * reward + self.discount * self.critic_target(next_state, next_target_action) - self.critic(state, action)) ** 2.0  # (batch_size,1)
			else:
				td_err_squared = (reward + self.discount * self.critic_target(next_state, next_target_action) - self.critic(state, action)) ** 2.0 #(batch_size,1)
			td_err_squared = torch.squeeze(td_err_squared, dim=1) #(batch_size,)
			X_v = torch.sum(td_err_squared * grad_l2_squared / behav_pol_den_target.clamp(min=self.clip_behav_den_val_min), dim=0)  # (batch_size,)

			if i_train_step % self.n_eval_freq == 0 and wandb_result!=None:
				grad_l2_squared = grad_l2_squared.cpu().numpy()
				td_err_squared = td_err_squared.cpu().numpy()

				wandb_result.update({"grad_Q_wrt_params_l2_squared": wandb.Histogram(grad_l2_squared),
									 "grad_Q_wrt_params_l2_squared/mean": grad_l2_squared.mean(),
									 "grad_Q_wrt_params_l2_squared/max": grad_l2_squared.max(),
									 "grad_Q_wrt_params_l2_squared/min": grad_l2_squared.min(),
									 "grad_Q_wrt_params_l2_squared/median": np.median(grad_l2_squared),
									 "td_err_squared_target_action": wandb.Histogram(td_err_squared),
									 "td_err_squared_target_action/mean": td_err_squared.mean(),
									 "td_err_squared_target_action/max": td_err_squared.max(),
									 "td_err_squared_target_action/min": td_err_squared.min(),
									 "td_err_squared_target_action/median": np.median(td_err_squared),
									 })

		return X_b, X_v, wandb_result



	def get_L(self, B, alpha=1.0, EPS=10.0 ** (-8.0), reg_multiplier=0.1):

		"""
		B: Hessian matrix of reward w.r.t. action (action dim x action dim)
		EPS: small positive number for checking if the eigenvalues are 0
		reg_multiplier = -2.0 # for gamma scale
			returns: transformation matrix L(s) (A(s)=L(s)L(s)^\top)
		"""

		B = (B + torch.transpose(B, dim0=1, dim1=2)) / 2.0
		B_eigval, B_eigvec = torch.linalg.eigh(B) # eigenvalues returned in ascending order
		act_dim = B.shape[-1]

		B_eigval_pos_mask = B_eigval > EPS  # (bs, act_dim)
		B_eigval_neg_mask = B_eigval < -EPS
		d_pos = torch.sum(B_eigval_pos_mask, dim=1)  # (bs, ). d_pos for each batch dim
		d_neg = torch.sum(B_eigval_neg_mask, dim=1)

		M_tilde = (B_eigval_pos_mask * torch.unsqueeze(d_pos, dim=1) - B_eigval_neg_mask * torch.unsqueeze(d_neg, dim=1)) * B_eigval
		M_tilde = M_tilde + torch.unsqueeze(torch.max(M_tilde, dim=1)[0] * reg_multiplier, dim=1)
		M_tilde = M_tilde * alpha + (1-alpha)*torch.ones(M_tilde.shape, device=self.device)
		M_tilde = torch.diag_embed(M_tilde)
		L_hat = B_eigvec @ (M_tilde ** (0.5))

		## using log trick for numerical stability
		logdet_M_tilde = (L_hat @ torch.transpose(L_hat, dim0=1, dim1=2)).logdet()
		L_hat = L_hat * torch.reshape(((-1 / (2. * act_dim)) * logdet_M_tilde).exp(), (-1, 1, 1))

		return L_hat, B_eigval


	def get_h_L(self, replay_buffer, next_target_action, h, L, learn_h, learn_L, behav_pol_den_target, i_train_step, n_value_updates, hessian, wandb_result=None, alpha=1.0):

		state  = replay_buffer.state_scaler.transform(replay_buffer.state[:replay_buffer.size]) ## buffer['state']
		action = replay_buffer.action[:replay_buffer.size]
		reward = replay_buffer.reward[:replay_buffer.size]
		next_state = replay_buffer.state_scaler.transform(replay_buffer.next_state[:replay_buffer.size])

		n = state.shape[0]

		if not learn_L and i_train_step==0:
			# if learn_h and learn_L are false, use the given h and idnetity metric
			L = torch.ones( n, self.action_dim, device=self.device)
			L = torch.diag_embed(L)

		if learn_h or learn_L:
			self.critic.zero_grad()

			# run in mini-batch
			n_batch = n // self.hessian_batch_size
			if n % self.hessian_batch_size !=0:
				n_batch = n_batch + 1

			B_eigval = torch.empty(n, self.action_dim, device=self.device)

			# for computing h
			if hessian is None:
				hessian = torch.empty(n, self.action_dim, self.action_dim, device=self.device)

			if i_train_step % n_value_updates == 0:
				# Hessian is updated at each hard critic target update (n_value_updates)
				# need to compute Hessian for initial timestep to compute init h
				for i_batch in range(n_batch):
					start_idx = i_batch * self.hessian_batch_size
					end_idx = (i_batch + 1) * self.hessian_batch_size
					batch_next_state           =         next_state[start_idx:end_idx]
					batch_next_target_action   = next_target_action[start_idx:end_idx]
					batch_B = get_critic_hess_action(self.critic_target, batch_next_state, batch_next_target_action)  # shape: (batch_size, act_dim, act_dim)
					hessian[start_idx:end_idx] = batch_B.detach()

					if learn_L and i_train_step !=0:
						with torch.no_grad():
							L_batch, B_eigval_batch = self.get_L(batch_B, alpha=alpha, EPS=10 ** (-8), reg_multiplier=self.reg_multiplier)
						L[start_idx:end_idx] = L_batch
						B_eigval[start_idx:end_idx] = B_eigval_batch

				if learn_L and i_train_step !=0:
					print(" ====  L updated ====")

					if i_train_step % self.n_eval_freq==0 and wandb_result!=None:
						# for debugging eigenvalues
						B_eigval = np.abs(B_eigval.cpu().detach().numpy())

						for i_dim in range(B_eigval.shape[1]):
							wandb_result[f"eigval_{i_dim}"] = wandb.Histogram(B_eigval[:, i_dim])
							wandb_result[f"eigval_{i_dim}/max"] = B_eigval[:, i_dim].max()
							wandb_result[f"eigval_{i_dim}/min"] = B_eigval[:, i_dim].min()
							wandb_result[f"eigval_{i_dim}/mean"] = B_eigval[:, i_dim].mean()
							wandb_result[f"eigval_{i_dim}/median"] = np.median(B_eigval[:, i_dim])

						max_eig_vals = np.amax(B_eigval, axis=1)
						second_max_eig_vals = np.partition(B_eigval, -2, axis=1)[:, -2]
						ratios = max_eig_vals / second_max_eig_vals

						wandb_result[f"eigval_ratio"] = wandb.Histogram(ratios)
						wandb_result[f"eigval_ratio/max"] = ratios.max()
						wandb_result[f"eigval_ratio/min"] = ratios.min()
						wandb_result[f"eigval_ratio/mean"] = ratios.mean()
						wandb_result[f"eigval_ratio/median"] = np.median(ratios)

			if learn_h: # h is updated at each step
				# compute h on a minibatch
				indices = np.random.randint(n, size=self.h_batch_size)

				tr_B = hessian.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)  # (data_size,)
				tr_B = tr_B.detach()

				X_b, X_v, wandb_result = self.get_h(tr_B[indices],
										  state[indices],
										  action[indices],
										  reward[indices],
										  next_state[indices],
										  next_target_action[indices],
										  behav_pol_den_target[indices],
										  i_train_step,
										  wandb_result)

				C_b = self.discount**2.0 / 4.0 * torch.sum((X_b/self.h_batch_size) ** 2.0)
				R_K = (4.0 * torch.pi) ** (- self.action_dim / 2.0)
				C_v = R_K * X_v / self.h_batch_size
				h = ((self.action_dim * C_v) / (4 * n * C_b)) ** (1.0 / (self.action_dim + 4.0))

				# for wamdb debugging
				if i_train_step % self.n_eval_freq == 0  and wandb_result!=None:
					tr_B = tr_B.cpu().numpy()

					wandb_result.update({"C_b":C_b.detach().cpu().numpy(),
								   "C_v": C_v.detach().cpu().numpy(),
								   "laplacian_Q_wrt_a'": wandb.Histogram(tr_B),
								   "laplacian_Q_wrt_a'/mean": tr_B.mean(),
								   "laplacian_Q_wrt_a'/max": tr_B.max(),
								   "laplacian_Q_wrt_a'/min": tr_B.min(),
								   "laplacian_Q_wrt_a'/median": np.median(tr_B),
									"bandwidth": h,
					})

		return h, L, hessian, wandb_result


	# # Dimension-wise clipping of the IS ratio
	def get_is_ratio(self, replay_buffer, next_target_action, h, L, learn_h, learn_L, behav_pol_den, behav_pol_den_target, i_train_step, n_value_updates, hessian, wandb_result=None, alpha=1.0):

		h, L, hessian, wandb_result = self.get_h_L(replay_buffer, next_target_action, h, L, learn_h, learn_L, behav_pol_den_target, i_train_step, n_value_updates, hessian, wandb_result, alpha)
		next_action = replay_buffer.next_action[:replay_buffer.size]		## action is not scaled
		with torch.no_grad():
			batch_kernel_input = torch.transpose(L, 1,2) @ torch.unsqueeze( (next_target_action - next_action), dim=2) / h  # (batch_sizes, act_dim, 1)
			batch_kernel_input = torch.squeeze(batch_kernel_input, dim=2) # (batch_sizes, act_dim)

		def wandb_logging(is_ratio, wandb_result):
			is_ratio_detached_np = np.squeeze(is_ratio.cpu().numpy())
			ess = np.sum(is_ratio_detached_np) ** 2.0 / np.sum(is_ratio_detached_np ** 2.0)

			wandb_result.update({
				"ess": ess,
				"is_ratio": wandb.Histogram(is_ratio_detached_np),
				"is_ratio/max": is_ratio_detached_np.max(),
				"is_ratio/min": is_ratio_detached_np.min(),
				"is_ratio/mean": is_ratio_detached_np.mean(),
				"is_ratio/median": np.median(is_ratio_detached_np),
			})
			return wandb_result

		with torch.no_grad():
			if not self.dim_wise_is_clip:
				kernel_val = functorch.vmap(gaussian_kernel)(batch_kernel_input)  # shape: (batch_size,)
				kernel_val = h ** (-self.action_dim) * kernel_val
				is_ratio = kernel_val / behav_pol_den
				is_ratio = torch.clamp(is_ratio, min=self.clip_val_min, max=self.clip_val_max)  # shape: (batch_size,)

			else:
				assert self.random == 0.0
				kernel_val = functorch.vmap(gaussian_kernel_dim_wise)(batch_kernel_input)  # shape: (batch_size,act_dim)
				kernel_val = kernel_val / h
				# clip dim-wise
				is_ratio = torch.clamp(kernel_val / behav_pol_den, min=self.clip_val_min,
									   max=self.clip_val_max)  # shape: (batch_size, act_dim)
				is_ratio = torch.prod(is_ratio, dim=1)

			# wandb logging after the clipping
			if i_train_step%self.n_eval_freq==0 and wandb_result!=None:
				wandb_result = wandb_logging(is_ratio, wandb_result)

		assert len(is_ratio.shape) == 1
		return is_ratio, hessian, L, wandb_result


	def eval_policy(self, replay_buffer, policy):
		start_state = replay_buffer.sample_start_state(scaled=True)
		start_state_unscaled = replay_buffer.state_scaler.inverse_transform(start_state)

		if self.env_is_ant:
			start_state_unscaled = torch.hstack([start_state_unscaled, torch.zeros([start_state_unscaled.shape[0], int(111 - 27)], device=self.device)])

		start_action = policy(start_state_unscaled)
		if self.normalized_action_value:
			R = self.critic(start_state, start_action).mean()
		else:
			R = (1. - self.discount) * self.critic(start_state, start_action).mean()
		R = R.detach().cpu()
		return float(R)


	def save_critic_target(self, filename):
		torch.save(self.critic_target.state_dict(), filename)
		# torch.save(self.mdp_nn_optimizer.state_dict(), filename + "_mdp_optimizer")


	def load_critic_target(self, filename):
		self.critic_target.load_state_dict(torch.load(filename, map_location=self.device))
		# self.mdp_nn_optimizer.load_state_dict(torch.load(filename + "_mdp_optimizer", map_location=self.device))


	def save_critic(self, filename):
		torch.save(self.critic.state_dict(), filename)
		# torch.save(self.mdp_nn_optimizer.state_dict(), filename + "_mdp_optimizer")


	def load_critic(self, filename):
		self.critic.load_state_dict(torch.load(filename, map_location=self.device))
		# self.mdp_nn_optimizer.load_state_dict(torch.load(filename + "_mdp_optimizer", map_location=self.device))


class KMIFQE_resample_TD(KMIFQE):
	def __init__(
			self,
			state_dim,
			action_dim,
			max_action,
			last_layer_activation = None,
			hidden_units=256,
			discount=0.99,
			tau=0.005,
			clip_behav_den_val_min = 0.0,
			dim_wise_is_clip=False,
			critic_target_path="./critic_targets/critic_target",
			normalized_action_value = False,
			relax_target_std=0.0,
			random=0.0,
			behav_bias=0.0,
			behav_std=0.2,
			clip_val_max=1e8,
			clip_val_min=1e-8,
			batch_size = 256,
			hessian_batch_size=256,
			h_batch_size=256,
			reg_multiplier = 0.1,
			weight_decay = 0.0,
			max_episode_len=None,
			n_eval_freq = 1000,
			env_is_ant = False,
			device='cpu'
	):
		super().__init__(state_dim,
			            action_dim,
						max_action,
						last_layer_activation,
						hidden_units,
						discount,
						tau,
						clip_behav_den_val_min,
						dim_wise_is_clip,
						critic_target_path,
						normalized_action_value,
						relax_target_std,
						random,
						behav_bias,
						behav_std,
						clip_val_max,
						clip_val_min,
						batch_size,
						hessian_batch_size,
						h_batch_size,
						reg_multiplier,
						weight_decay,
						max_episode_len,
						n_eval_freq,
						env_is_ant,
						device)

		self.device = torch.device(device)
		self.last_layer_activation = last_layer_activation
		self.critic = Critic(state_dim, action_dim, hidden_units, last_layer_activation).to(self.device)
		self.critic_target = copy.deepcopy(self.critic)
		self.weight_decay = weight_decay
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4, weight_decay=self.weight_decay)
		self.clip_behav_den_val_min = clip_behav_den_val_min
		self.dim_wise_is_clip = dim_wise_is_clip
		self.critic_target_path = critic_target_path
		self.normalized_action_value = normalized_action_value

		self.discount = discount
		self.tau = tau

		self.total_it = 0

		self.hidden_units = hidden_units
		self.action_dim = action_dim
		self.max_action = max_action
		self.relax_target_std = relax_target_std
		self.random = random
		self.behav_bias = behav_bias
		self.behav_std = behav_std
		self.clip_val_max = clip_val_max
		self.clip_val_min = clip_val_min
		self.batch_size = batch_size
		self.hessian_batch_size = hessian_batch_size
		self.h_batch_size = h_batch_size
		self.reg_multiplier = reg_multiplier
		self.max_episode_len = max_episode_len

		self.env_is_ant = env_is_ant

	def train_OPE(self, replay_buffer, policy, behav_policy, h, i_train_step, hessian, L, behav_pol_den, behav_pol_den_target, next_target_action, n_value_updates, wandb_result=None, learn_h=False, learn_L=False, alpha=1.0):
		# policy: deterministic target policy
		# behav_policy: deterministic policy for making the data collecting (behavior) policy

		if i_train_step == 0:
			# For initial h and L
			with torch.no_grad():
				learn_L = False
				next_action = replay_buffer.next_action[:replay_buffer.size]		## action is not scaled
				next_state_unscaled = replay_buffer.next_state[:replay_buffer.size]	## will be passed to policy, no need to scale

				if self.env_is_ant:
					next_state_unscaled = torch.hstack([next_state_unscaled, torch.zeros([next_state_unscaled.shape[0], int(111 - 27)], device=self.device)])

				next_target_action = policy(next_state_unscaled)
				next_behav_action_mean = behav_policy(next_state_unscaled)
				
				if not self.dim_wise_is_clip:
					## (bs,)
					behav_pol_den = self.get_behav_pol_den(
						action	= next_action,
						mean	= next_behav_action_mean + self.behav_bias
					)
				else:
					## (bs, act_dim)
					behav_pol_den = self.get_gauss_behav_pol_den_dim_wise(
						action	= next_action,
						mean	= next_behav_action_mean + self.behav_bias
					)
				
				## (bs,) used for bandwidth update, not IS ratio
				behav_pol_den_target = self.get_behav_pol_den(
					action	= next_target_action,
					mean	= next_behav_action_mean + self.behav_bias
				)

				del next_behav_action_mean

		# C_v, C_b are updated every step when the critic is updated, thus h is updated at each step, therefore the IS ratio is updated at each step
		is_ratio, hessian, L, wandb_result = self.get_is_ratio(replay_buffer, next_target_action, h, L, learn_h, learn_L, behav_pol_den, behav_pol_den_target, i_train_step, n_value_updates, hessian, wandb_result, alpha)

		# update Q
		with torch.no_grad():
			resample_prob = is_ratio / torch.sum(is_ratio)
			is_ratio_avg = torch.mean(is_ratio)
			resample_idx = torch.multinomial(resample_prob, self.batch_size, replacement=True)

		batch = replay_buffer.sample(self.batch_size, scaled=True, resample_idx=resample_idx)

		state       = batch['state']
		action      = batch['action']
		next_state  = batch['next_state']
		next_action = batch['next_action']
		reward      = batch['reward']
		not_done    = batch['not_done']

		with torch.no_grad():
			critic_target = self.critic_target(next_state, next_action)
			if self.normalized_action_value:
				TD_target = (1.0-self.discount) * reward + self.discount * not_done * critic_target
			else:
				TD_target = reward + self.discount * not_done * critic_target

		# TD_error = (TD_target - self.critic(state, action))
		current_Q = self.critic(state, action)



		critic_loss = is_ratio_avg * F.mse_loss(current_Q, TD_target)


		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()



		###############################################################################################################

		if ((i_train_step+1) % n_value_updates == 0) or i_train_step==0:

			# update critic target
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				# target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data) # soft update
				target_param.data.copy_(param.data) # hard update

		# for wandb logging
		if i_train_step%self.n_eval_freq ==0 and wandb_result!=None:
			critic_loss = critic_loss.detach().cpu().numpy()
			print("########################################")
			print(f"critic_loss: {critic_loss:f}")

			resample_idx = resample_idx.detach().cpu().numpy()
			num_init_states_in_minibatch  = (resample_idx % int(self.max_episode_len) == 0).sum().item()
			current_Q = np.squeeze(current_Q.detach().cpu().numpy())
			TD_target = np.squeeze(TD_target.detach().cpu().numpy())
			unique_resample_idx = np.unique(resample_idx)
			behav_pol_den_target_np = behav_pol_den_target.detach().cpu().numpy()

			wandb_result.update({"num_init_states_in_minibatch": num_init_states_in_minibatch,
								 "critic_loss": critic_loss,
								 "current_Q": wandb.Histogram(current_Q),
								 "current_Q/max": current_Q.max(),
								 "current_Q/min": current_Q.min(),
								 "current_Q/mean": current_Q.mean(),
								 "current_Q/median": np.median(current_Q),
								 "TD_target": wandb.Histogram(TD_target),
								 "TD_target/max": TD_target.max(),
								 "TD_target/min": TD_target.min(),
								 "TD_target/mean": TD_target.mean(),
								 "TD_target/median": np.median(TD_target),
								 "unique_resample_idx/max": unique_resample_idx.max(),
								 "unique_resample_idx/min": unique_resample_idx.min(),
								 "unique_resample_idx/mean": unique_resample_idx.mean(),
								 "unique_resample_idx/median": np.median(unique_resample_idx),
								 "num_unique_resample_idx": len(unique_resample_idx),
								 "unique_resample_idx": wandb.Histogram(unique_resample_idx),
									"behav_pol_den_target": wandb.Histogram(behav_pol_den_target_np),
									"behav_pol_den_target/max": behav_pol_den_target_np.max(),
									"behav_pol_den_target/min": behav_pol_den_target_np.min(),
									"behav_pol_den_target/mean": behav_pol_den_target_np.mean(),
									"behav_pol_den_target/median": np.median(behav_pol_den_target_np),
			})

		return hessian, L, behav_pol_den, behav_pol_den_target, next_target_action, wandb_result



