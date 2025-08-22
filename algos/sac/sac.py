from copy import deepcopy
from dataclasses import dataclass, asdict
import itertools
import os

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

import algos.sac.core as core
from algos.sac.utils import *
from algos.utils.logx import EpochLogger

import mani_skill.envs
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
import gymnasium as gym
import time
import wandb
import tyro


@dataclass
class Args:
    """ Configurations for SAC """
    seed: int = 69420
    record: bool = True
    save_train_video_freq: int = 1_000
    save_trajectory: bool = False
    wandb_log: bool = True
    wandb_project: str = "sac_maniskill"
    wandb_entity: str = "sman2"
    actor_critic = core.MLPActorCritic
    output_dir: str = 'runs/'
    """ RL - Params """
    total_steps: int = 10_000_000   # Total number of steps to run the environment for
    ckpt_every: int = 10_000        # Save checkpoint every X steps
    replay_size: int = 2_000_000    # Total RB length
    gamma: float = 0.99             # RL future reward discount factor
    polyak: float = 0.995           # Polyak-averaging for critic network parameters
    alpha0: float = 0.01             # Entropy temperature parameter
    """ RL - Env"""
    num_train_envs: int = 256       # Number of parallel environments
    num_test_envs: int = 6          # Number of test environments
    start_steps: int = 10_000       # Execute random action period
    max_ep_len: int = 100           # Max episode length
    control_mode: str = "pd_joint_delta_pos"
    """ RL - Update """
    update_after: int = 1_000       # Execute random policy until X samples
    u2d_ratio: float = 15./100       # Gradient steps per X new env samples
    test_every_updates: int = 1_000   # Policy eval every X updates
    """ Network configs """
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size: int = 512
    hidden_sizes: tuple[int,...] = (256,256,256)
    activation: ActivationType = ActivationType.RELU
    optimizer: OptimizerType = OptimizerType.ADAMW
    lr: float = 1e-3                # Network learning-rate


# TODO: add network initialization
# TODO: fix checkpoint
# TODO: rename

def sac(env_fn:str, args:Args):

    if args.wandb_log:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=asdict(args),
            name=f"sac_seed_{args.seed}_{time.strftime('%Y-%m-%d_%H-%M-%S')}",
            monitor_gym=True,
            save_code=True,
        )

    save_dir = args.output_dir + f"seed-{args.seed}_-_{time.strftime('%Y-%m-%d_%H-%M-%S')}/"
    eval_output_dir = save_dir + "videos/"
    os.makedirs(save_dir, exist_ok=True)
    logger_kwargs = dict(output_dir=save_dir, exp_name="sac")
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env: gym.Env
    test_env: gym.Env
    env_kwargs = dict(obs_mode="state", sim_backend="gpu", max_episode_steps=args.max_ep_len)
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    eval_env_kwargs = dict(**env_kwargs, render_mode="rgb_array") if args.record else env_kwargs
    env = ManiSkillVectorEnv(
        gym.make(env_fn, **env_kwargs, num_envs=args.num_train_envs), 
        auto_reset=True, 
        ignore_terminations=False)
    test_env = ManiSkillVectorEnv(
        gym.make(env_fn, **eval_env_kwargs, num_envs=args.num_test_envs), 
        auto_reset=False, 
        ignore_terminations=False)

    obs_dim = env.single_observation_space.shape[-1]
    act_dim = env.single_action_space.shape[0]
    act_limit = env.single_action_space.high[0]

    if args.record:
        test_env = RecordEpisode(
            test_env, 
            output_dir=eval_output_dir, 
            save_trajectory=args.save_trajectory, 
            save_video=args.record, 
            trajectory_name="trajectory", 
            max_steps_per_video=args.max_ep_len, 
            video_fps=30
            )

    activation: type[Module] = ACTIVATIONS[args.activation]
    optimizer: type[Optimizer] = OPTIMIZERS[args.optimizer]

    ac = args.actor_critic(obs_dim, act_dim, act_limit, args.hidden_sizes, activation).to(device=args.device)
    ac_targ = deepcopy(ac)
    # Freeze target network wrt to optimizers (onyl update by polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
    # Put Q-networks' params in a single list
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
    alpha = args.alpha0

    replay_buffer = ReplayBuffer(obs_dim, act_dim, args.replay_size, args.device)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log("Number of params: pi: %d, q1: %d, q2: %d"%var_counts)

    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q1 = ac.q1.forward(o,a)
        q2 = ac.q2.forward(o,a)
        # Bellman backup for Q
        with torch.no_grad():
            # Target acitons come from current policy
            a2, logp_a2 = ac.pi.forward(o2)
            # Target Q-values
            q1_pi_targ = ac_targ.q1.forward(o2, a2)
            q2_pi_targ = ac_targ.q2.forward(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup: Tensor = r + args.gamma * (1-d) * (q_pi_targ - alpha * logp_a2)

        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                      Q2Vals=q2.detach().cpu().numpy())

        return loss_q, q_info

    def compute_loss_pi(data):
        o = data['obs']
        pi, logp_pi = ac.pi.forward(o)
        q1_pi = ac.q1.forward(o, pi)
        q2_pi = ac.q2.forward(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        # Entropy regularization
        loss_pi = (alpha * logp_pi - q_pi).mean()
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())
        return loss_pi, pi_info
    
    # Set up optimizers 
    pi_optimizer = optimizer(ac.pi.parameters(), lr=args.lr)
    q_optimizer = optimizer(q_params, lr=args.lr)

    def update(data):
        # Gradient descent for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks to stop gradient computation during policy learning
        for p in q_params:
            p.requires_grad = False

        # Gradient ascent for pi
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        for p in q_params:
            p.requires_grad = True

        logger.store(LossPi=loss_pi.item(), **pi_info)

        if args.wandb_log:
            wandb.log({
                "Updates" : num_updates,
                "loss/Q_Loss": loss_q.item(),
                "loss/Pi_Loss": loss_pi.item(),
                "values/Q1_Vals_Mean": q_info['Q1Vals'].mean(),
                "values/LogPi_Mean": pi_info['LogPi'].mean()
            }, step=global_steps) # Use the global step 't' as the x-axis

        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # Using in-place operations like "mul_" and "add_" to not make new tensors
                p_targ.data.mul_(args.polyak)
                p_targ.data.add_((1-args.polyak)*p.data)

    def get_action(o, deterministic=False):
        return ac.act(o, deterministic)
    
    def test_agent():
        truncated: Tensor
        o, ep_ret, ep_len = test_env.reset()[0], 0., 0.
        d = truncated = torch.tensor([False], dtype=torch.bool)
        while not (d | truncated).all():
            # Take deterministic actions at test time
            o, r, d, truncated, _ = test_env.step(get_action(o, True))
            ep_ret += r
            ep_len += ~ (d | truncated)
        # logger.store(TestEpRet=ep_ret.cpu().numpy(), TestEpLen=ep_len)
        return {"test/return": torch.mean(ep_ret.cpu()), "test/length": torch.mean(ep_len.cpu())}

    total_steps = args.total_steps
    num_updates = 0
    global_steps = 0
    update_to_go = 0.0 # <-- Initialize the update budget
    start_time = time.time()
    next_test_at_update = args.test_every_updates 
    o, ep_ret, ep_len = env.reset()[0], 0, 0

    while global_steps < total_steps:
        global_steps += args.num_train_envs

        # Randomly sample actions until start_steps
        if global_steps > args.start_steps:
            a = get_action(o)
        else: 
            a = torch.tensor(env.action_space.sample())

        # Step the env
        o2, r, term, trunc, _ = env.step(a)
        # d = torch.logical_or(term, trunc)
        d = term
        # ep_ret += r
        # ep_len += ~d
        # d = False if ep_len==args.max_ep_len else d
        replay_buffer.store_batch(o, a, r, o2, d)
        # Update most recent observation
        o = o2

        # # End of traj handling
        # if d or (ep_len == args.max_ep_len):
        #     o, ep_ret, ep_len = env.reset()[0], 0, 0
        #     # logger.store(EpRet=ep_ret, EpLen=ep_len)

        # Updates and evals handling
        if global_steps >= args.update_after:
            update_to_go += args.num_train_envs * args.u2d_ratio
            num_updates_to_run = int(update_to_go)

            if num_updates_to_run > 0:
                for _ in range(num_updates_to_run):
                    batch = replay_buffer.sample_batch(args.batch_size)
                    update(batch) # Pass global_steps for logging
                    num_updates += 1
                
                update_to_go -= num_updates_to_run

            if num_updates >= next_test_at_update:
                print(f"\n--- Testing at Step: {global_steps}, Updates: {num_updates} ---")
                test_metrics = test_agent()
                print(f"Test Return = {test_metrics['test/return']:.2f}")
                print(f"Test Length = {test_metrics['test/length']:.2f}")

                if args.wandb_log:
                    # For test logs, use the number of updates as the x-axis
                    wandb.log(test_metrics, step=global_steps)
                    wandb.log({"Updates": num_updates}, step=global_steps)
                
                # Set the next testing waypoint
                next_test_at_update += args.test_every_updates

        # Save checkpoints
        if (global_steps+1) % args.ckpt_every == 0:
            checkpoint = {
                'steps': global_steps,
                'updates': num_updates,
                'model_state': ac.state_dict(),
                'target_state': ac_targ.state_dict(),
                'optimizer_state': {
                    'pi': pi_optimizer.state_dict(),
                    'q': q_optimizer.state_dict(),
                }
            }
            logger.save(checkpoint,global_steps, num_updates)


if __name__ == '__main__':
    args = tyro.cli(Args)
    
    sac("PickCube-v1", args)
