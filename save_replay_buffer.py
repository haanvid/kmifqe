import numpy as np
import torch
import gym
import argparse
import os
import replay_buffer
import TD3


def sample_action(policy, state, env, max_action, action_dim, random, behav_bias, behav_std):
    if  np.random.uniform(0, 1) < random:
        action = env.action_space.sample()
    else:
        action = policy.select_action(np.array(state)) + behav_bias + np.random.normal(0, max_action * behav_std, size=action_dim)
    return action


def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")
    parser.add_argument("--env", default="Hopper-v2")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005)
    parser.add_argument("--random", default=0.0, type=float)
    parser.add_argument("--behav_bias",      default=0.0, type=float)
    parser.add_argument("--behav_std", default=0.3, type=float)
    parser.add_argument("--policy_idx", default="medium", help="medium, expert", type=str)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", default=False, type=boolean)
    parser.add_argument("--buffers_dir", default="./data/buffers")
    parser.add_argument("--policy_dir", default="./data/policies")

    args = parser.parse_args()

    os.makedirs(os.path.join(args.buffers_dir, args.env), exist_ok=True)
    env = gym.make(args.env)

    # Set random seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # for TD3 policy
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "device": args.device
    }

    kwargs["policy_noise"] = 0.2 * max_action
    kwargs["noise_clip"] = 0.5 * max_action
    kwargs["policy_freq"] = 2

    # Load behavior policy
    policy = TD3.TD3(**kwargs)
    policy_path = f"{args.policy_dir}/{args.env}/{args.policy_idx}"
    policy.load(policy_path)
    print("Policy loaded from " + policy_path)

    replay_buffer = replay_buffer.ReplayBuffer(state_dim, action_dim, device=args.device)

    state, done = env.reset(), False
    action = sample_action(policy=policy, state=state, env=env, max_action=max_action, action_dim=action_dim, random=args.random, behav_bias=args.behav_bias,
                           behav_std=args.behav_std)
    replay_buffer.add_start(state)

    episode_ret = 0
    episode_discounted_ret = 0
    episode_timesteps = 0
    episode_num = 0

    # Collect data
    ret_list = []
    discounted_ret_list = []
    normalized_discounted_ret_list = []

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1
        next_state, reward, done, _ = env.step(action)
        next_action = sample_action(policy=policy, state=next_state, env=env, max_action=max_action, action_dim=action_dim, random=args.random, behav_bias=args.behav_bias, behav_std=args.behav_std)

        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        replay_buffer.add(state, action, next_state, next_action, reward, done_bool)

        state  = next_state
        action = next_action
        episode_ret += reward
        episode_discounted_ret += args.discount**(episode_timesteps-1.0) * reward

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            episode_normalized_discounted_ret =  (1.0 - args.discount) * episode_discounted_ret
            ret_list.append(episode_ret)
            discounted_ret_list.append(episode_discounted_ret)
            normalized_discounted_ret_list.append(episode_normalized_discounted_ret)
            if args.verbose:
                print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Noramlized Discounted Return: {episode_normalized_discounted_ret:.3f}")
            state, done = env.reset(), False
            action = sample_action(policy=policy, state=state, env=env, max_action=max_action, action_dim=action_dim, random=args.random, behav_bias=args.behav_bias, behav_std=args.behav_std)
            episode_ret = 0
            episode_discounted_ret = 0
            episode_timesteps = 0
            episode_num += 1
            replay_buffer.add_start(state)

    replay_buffer.fit_scalers()
    save_path = f"{args.buffers_dir}/{args.env}/{args.policy_idx}_{int(args.max_timesteps)}_rand{args.random:.1f}_bias{args.behav_bias:.1f}_std{args.behav_std:.1f}.pt"
    torch.save(replay_buffer, save_path)

    print("===========================================================================================================")
    print(f"saved replay buffer: {save_path}")
    print("Policy loaded from " + policy_path)
    print(f"Random: {args.random}")
    print(f"Bias:   {args.behav_bias}")
    print(f"Std:    {args.behav_std}")
    print(f"Env : {args.env}")
    print(f"Number of episodes : {episode_num}")
    print("-----------------------------------------------------------------------------------------------------------")
    print(f"avg      of undiscounted return over {episode_num} episodes : {np.mean(ret_list)}")
    print(f"stderror of undiscounted return over {episode_num} episodes : {np.std(ret_list) / np.sqrt(episode_num)}")
    print("-----------------------------------------------------------------------------------------------------------")
    print(f"avg      of gamma={args.discount} discounted return over {episode_num} episodes : {np.mean(discounted_ret_list)}")
    print(f"stderror of gamma={args.discount} discounted return over {episode_num} episodes : {np.std(discounted_ret_list) / np.sqrt(episode_num)}")
    print("-----------------------------------------------------------------------------------------------------------")
    print(f"avg      of  normalized discounted return over {episode_num} episodes : {np.mean(normalized_discounted_ret_list)}")
    print(f"stderror of  normalized discounted return over {episode_num} episodes : {np.std(normalized_discounted_ret_list)/np.sqrt(episode_num)}")
    print("===========================================================================================================")
