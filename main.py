import numpy as np
import torch
import gym
import argparse
import os
import KMIFQE
import TD3
import utils
import copy



if __name__ == "__main__":

	def boolean(v):
		if isinstance(v, bool):
			return v
		if v.lower() in ('yes', 'true', 't', 'y', '1'):
			return True
		elif v.lower() in ('no', 'false', 'f', 'n', '0'):
			return False
		else:
			raise argparse.ArgumentTypeError('Boolean value expected.')


	parser = argparse.ArgumentParser()
	parser.add_argument("--algo", default="KMIFQE_resample_TD", help="KMIFQE_resample_TD", type=str)
	parser.add_argument("--env", default="Hopper-v2", help="HalfCheetah-v2, Hopper-v2, Walker2d-v2", type=str)
	parser.add_argument("--buffer_size", default=int(1e6), type=int)
	parser.add_argument("--target_policy_idx", default="expert", help="medium, expert", type=str)
	parser.add_argument("--behavior_policy_idx", default="medium", help="medium, expert", type=str)
	parser.add_argument("--random", default=0.0, type=float)
	parser.add_argument("--behav_bias", default=0.0, type=float)
	parser.add_argument("--behav_std", default=0.3, type=float)
	parser.add_argument("--n_train_steps", default=int(1e6), type=int)
	parser.add_argument("--batch_size", default=2048, type=int)
	parser.add_argument("--hessian_batch_size", default=5000, type=int)
	parser.add_argument("--h_batch_size", default=2048, type=int)
	parser.add_argument("--reg_multiplier", default=0.1, type=float)
	parser.add_argument("--last_layer_activation", default="None", help="exp, None", type=str)
	parser.add_argument("--hidden_units", default=256, type=int)
	parser.add_argument("--n_value_updates", default=int(1e3), type=int)
	parser.add_argument("--discount", default=0.99, type=float)
	parser.add_argument("--tau", default=0.005, type=float)
	parser.add_argument("--relax_target_std", default=0.0, type=float)
	parser.add_argument("--n_eval_freq", default=int(1e3), type=int)
	parser.add_argument("--device", default="cpu", type=str)
	parser.add_argument("--dim_wise_is_clip", default=True, type=boolean)
	parser.add_argument("--clip_val_max", default=2.0, type=float)
	parser.add_argument("--clip_val_min", default=0.001, type=float)
	parser.add_argument("--clip_behav_den_val_min", default=1e-5, type=float)
	parser.add_argument("--learn_h", default=True, type=boolean)
	parser.add_argument("--fixed_h", default=0.2, type=float)
	parser.add_argument("--learn_L", default=True, type=boolean)
	parser.add_argument("--alpha", default=1.0, type=float)
	parser.add_argument("--weight_decay", default=0.0, type=float)
	parser.add_argument("--normalized_action_value", default=True, type=boolean)
	parser.add_argument("--seed", default=0, type=int)

	parser.add_argument("--buffers_dir", default="./data/buffers", type=str)
	parser.add_argument("--policies_dir",  default="./data/policies", type=str)
	parser.add_argument("--evals_dir",       default="./results/evals", type=str)

	args = parser.parse_args()
	config = vars(args)

	# make dir for the result
	results_dir_dict = copy.deepcopy(config)
	keys_to_delete = [ 'n_eval_freq','seed',
		               'buffers_dir', 'policies_dir', 'evals_dir',
					   'device', 'hessian_batch_size', 'n_train_steps']

	results_dir_dict = {key: value for key, value in results_dir_dict.items() if key not in keys_to_delete}
	evals_dict_dir   = utils.make_dir_with_dict(results_dir_dict, start_dir=args.evals_dir)
	eval_path        = os.path.join(evals_dict_dir, f"seed_{args.seed:d}.csv")


	gt_policy_val_dict = {
		"HalfCheetah-v2" 	:  7.267,
		"Hopper-v2"		 	:  2.571,
		"Walker2d-v2"	 	:  2.693,
		"Pendulum-v0"       : -3.791,
	}

	env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim  = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"hidden_units": args.hidden_units,
		"discount": args.discount,
		"tau": args.tau,
		"relax_target_std": args.relax_target_std,
		"device": args.device
	}

	KMIFQE_kwargs = copy.deepcopy(kwargs)
	KMIFQE_kwargs.update({
		"last_layer_activation":args.last_layer_activation,
		"clip_behav_den_val_min":args.clip_behav_den_val_min,
		"dim_wise_is_clip": args.dim_wise_is_clip,
		"random": args.random,
		"behav_bias": args.behav_bias,
		"behav_std": args.behav_std,
		"clip_val_max": args.clip_val_max,
		"clip_val_min": args.clip_val_min,
		"batch_size": args.batch_size,
		"hessian_batch_size": args.hessian_batch_size,
		"h_batch_size": args.h_batch_size,
		"reg_multiplier":args.reg_multiplier,
		"weight_decay":args.weight_decay,
		"max_episode_len":env._max_episode_steps,
		"normalized_action_value":args.normalized_action_value,
		"n_eval_freq": args.n_eval_freq
	})

	ope = KMIFQE.KMIFQE_resample_TD(**KMIFQE_kwargs)
	env = gym.make(args.env)

	# load data
	buffer_path = f"{args.buffers_dir}/{args.env}/{args.behavior_policy_idx}_{int(args.buffer_size):d}_rand{args.random:.1f}_bias{args.behav_bias:.1f}_std{args.behav_std:.1f}.pt"
	replay_buffer = torch.load(buffer_path, map_location=torch.device(args.device))
	replay_buffer.set_device(args.device)

	# Load policy
	del kwargs['relax_target_std']
	del kwargs['hidden_units']
	kwargs['action_dim'] = env.action_space.shape[0]
	del env

	# target policy
	policy = TD3.TD3(**kwargs)
	policy_path = f"{args.policies_dir}/{args.env}/{args.target_policy_idx}"
	policy.load(policy_path)

	# behavior policy
	behav_policy = TD3.TD3(**kwargs)
	behav_policy_path = f"{args.policies_dir}/{args.env}/{args.behavior_policy_idx}"
	behav_policy.load(behav_policy_path)

	print("Train KMIFQE")

	h = args.fixed_h
	L = None
	hessian=None
	behav_pol_den=None
	behav_pol_den_target=None
	next_target_action = None


	for i_train_step in range(int(args.n_train_steps+1)):

		hessian, L, behav_pol_den, behav_pol_den_target, next_target_action\
								= ope.train_OPE(replay_buffer		= replay_buffer,
												policy				= policy.actor,
												behav_policy		= behav_policy.actor,
												h					= h,
												L				    = L,
												i_train_step		= i_train_step,
												hessian 			= hessian,
												behav_pol_den       = behav_pol_den,
												behav_pol_den_target= behav_pol_den_target,
												next_target_action  = next_target_action,
												n_value_updates		= args.n_value_updates,
												learn_h				= args.learn_h,
												learn_L				= args.learn_L,
												alpha				= args.alpha,
												)

	
		if i_train_step % args.n_eval_freq == 0:

			print("i_train_step", i_train_step)
			eval = ope.eval_policy(replay_buffer, policy.actor)
			print(f"eval : {eval:.3f}")

			squared_err = (gt_policy_val_dict[args.env] - eval) ** (2.0)
			gt_policy_val = gt_policy_val_dict[args.env]

			utils.write_csv(i_train_step, squared_err, eval, gt_policy_val, eval_path)

