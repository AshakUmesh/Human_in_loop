'''
Training script for Human-in-the-Loop Deep RL with CARLA 0.9.15.
Mouse + Keyboard replaces the G29 steering wheel.

Controls during training:
  Hold RIGHT MOUSE BUTTON   → activate human takeover
  Move Mouse Left/Right     → analog steering
  A / Left Arrow            → steer left (keyboard, additive)
  D / Right Arrow           → steer right (keyboard, additive)
  S / Down Arrow            → centering steer
  Release RMB               → return control to AI
'''

import numpy as np
import time
import scipy.io as scio
import os
import pygame
import signal
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./TD3_based_DRL/checkpoints/log')

# CARLA environment (CARLA 0.9.15 + mouse/keyboard)
from env import scenario

# Utilities
from utils import set_seed, signal_handler, get_path, RND


def RL_training():

    set_seed(args.seed)

    # ── Select algorithm ──────────────────────────────────────
    if args.algorithm == 0:
        from TD3_based_DRL.TD3HUG import DRL
        log_dir = 'TD3_based_DRL/checkpoints/TD3HUG.pth'
    elif args.algorithm == 1:
        from TD3_based_DRL.TD3IARL import DRL
        log_dir = 'TD3_based_DRL/checkpoints/TD3IARL.pth'
    elif args.algorithm == 2:
        from TD3_based_DRL.TD3HIRL import DRL
        log_dir = 'TD3_based_DRL/checkpoints/TD3HIRL.pth'
    else:
        from TD3_based_DRL.TD3 import DRL
        log_dir = 'TD3_based_DRL/checkpoints/TD3.pth'

    # ── Build env + agent ─────────────────────────────────────
    env   = scenario()
    s_dim = [env.observation_size_width, env.observation_size_height]
    a_dim = env.action_size
    DRL   = DRL(a_dim, s_dim)

    if args.reward_shaping == 3:
        rnd = RND()

    if args.resume and os.path.exists(log_dir):
        checkpoint  = torch.load(log_dir)
        DRL.load(log_dir)
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    exploration_rate = args.initial_exploration_rate

    # ── Metrics ───────────────────────────────────────────────
    total_step = 0
    a_loss, c_loss = 0, 0
    loss_critic, loss_actor = [], []

    episode_reward_list, global_reward_list, episode_duration_list = [], [], []

    previous_action = [[] for _ in range(int(args.maximum_episode))]
    final_action    = [[] for _ in range(int(args.maximum_episode))]

    reward_i_record = [[] for _ in range(int(args.maximum_episode))]
    reward_e_record = [[] for _ in range(int(args.maximum_episode))]

    action_disturbing_degree = []
    intervene_percent_per_episode = [[] for _ in range(int(args.maximum_episode))]
    intervene_percent = []

    x_per_episode = [[] for _ in range(int(args.maximum_episode))]
    y_per_episode = [[] for _ in range(int(args.maximum_episode))]

    qlist = []

    path_generator = get_path()
    start_time     = time.perf_counter()

    # ── Training loop ─────────────────────────────────────────
    for i in range(start_epoch, int(args.maximum_episode)):
        reward    = 0
        ep_reward = 0
        step      = 0
        step_intervene = 0
        done      = False

        list_fdbk = [None]   # list_fdbk[0] = None by design

        flag_qrecord    = 0
        pid_activation  = 0
        pid_seed        = np.random.randint(0, 3)
        pid_intergal_value = 0

        State, scope = env.restart()

        while True:
            # ── AI action ──────────────────────────────────────
            action = DRL.choose_action(State)
            action = np.clip(np.random.normal(action, exploration_rate), -1, 1)
            previous_action[i].append(action)

            # ── PI controller guidance (optional) ──────────────
            if args.pid_controller_guidance:
                ego_y   = env.ego_vehicle.get_location().y
                ego_x   = env.ego_vehicle.get_location().x
                ego_yaw = env.ego_vehicle.get_transform().rotation.yaw
                threshold  = 10 if (205 < ego_y < 215) or (230 < ego_y < 240) else 1
                colli_risk = (abs(ego_x - path_generator(
                    np.clip(ego_y, 200, 250))) > threshold)
                left_risk  = (ego_x > 338.5) and (ego_yaw < 90)
                right_risk = (ego_x < 335)   and (ego_yaw > 90)

                if not pid_activation:
                    pid_intergal_value = 0

                if (colli_risk or left_risk or right_risk) and (step != 0) and (i % 3 == pid_seed):
                    pid_activation = True
                    xreal = scope['position_x']
                    xref  = path_generator(np.clip(scope['position_y'], 200, 250))
                    pid_intergal_value += (xreal - xref)
                    action = np.clip(0.3 * (xreal - xref) + 0.0 * pid_intergal_value, -1, 1)
                else:
                    pid_activation = False

            # ── Environment step ───────────────────────────────
            # env.run_step returns human_control=None when AI drives,
            # float when human is holding RMB
            State_, action_fdbk, reward_e, _, done, scope = env.run_step(action)
            list_fdbk.append(action_fdbk)

            # ── Reward shaping ─────────────────────────────────
            if args.reward_shaping == 1:
                if (action_fdbk is not None) or (pid_activation is True):
                    if step_intervene == 0:
                        reward_i = -10
                        step_intervene += 1
                    else:
                        reward_i = 0
                else:
                    reward_i = 0
                    step_intervene = 0
            elif args.reward_shaping == 2:
                ego_y    = env.ego_vehicle.get_location().y
                reward_i = 250 - ego_y
            elif args.reward_shaping == 3:
                error, mu, std = rnd.forward(State_)
                reward_i = (min(max(1 + (error - mu) / std, 0.35), 2) - 0.35) * 10
            else:
                reward_i = 0

            reward = reward_e + reward_i

            # ── Replay buffer storage ─────────────────────────
            if action_fdbk is not None:
                # Human intervened via mouse/keyboard
                if (flag_qrecord == 0) and (list_fdbk[-2] is None):
                    bs = torch.tensor(State, dtype=torch.float).view(
                        1, env.observation_size_height,
                        env.observation_size_width).to(args.device)
                    ba = torch.tensor(DRL.actor(bs), dtype=torch.float).to(args.device)
                    q1, q2 = DRL.critic([bs, ba])
                    qlist.append([q1.detach().cpu().numpy(),
                                  q2.detach().cpu().numpy()])

                action_disturbing_degree.append([action_fdbk, float(action)])
                intervene_percent_per_episode[i].append(action_fdbk)
                action      = action_fdbk
                intervention = 1
                DRL.store_transition(State, action, action_fdbk, intervention, reward, State_)

            elif pid_activation is True:
                intervention = 1
                if flag_qrecord == 0:
                    bs = torch.tensor(State, dtype=torch.float).view(
                        1, env.observation_size_height,
                        env.observation_size_width).to(args.device)
                    ba = torch.tensor(action, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(args.device)
                    q1, q2 = DRL.critic([bs, ba])
                    qlist.append([q1.detach().cpu().numpy(),
                                  q2.detach().cpu().numpy()])
                    flag_qrecord = 1
                intervene_percent_per_episode[i].append(action)
                DRL.store_transition(State, action, action_fdbk, intervention, reward, State_)

            else:
                intervention = 0
                DRL.store_transition(State, action, action_fdbk, intervention, reward, State_)

            # ── Learning step ──────────────────────────────────
            learn_threshold = args.warmup_threshold if args.warmup else 256
            if total_step > learn_threshold:
                c_loss, a_loss = DRL.learn(epoch=i)
                loss_critic.append(np.average(c_loss))
                loss_actor.append(np.average(a_loss))
                exploration_rate = (exploration_rate * args.exploration_decay_rate
                                    if exploration_rate > args.cutoff_exploration_rate
                                    else 0.05)

            # ── Bookkeeping ────────────────────────────────────
            ep_reward  += reward
            global_reward_list.append([reward_e, reward_i])
            reward_e_record[i].append(reward_e)
            reward_i_record[i].append(reward_i)
            final_action[i].append(action)
            x_per_episode[i].append(scope['position_x'])
            y_per_episode[i].append(scope['position_y'])

            State       = State_
            total_step += 1
            step       += 1

            if done:
                mean_reward = ep_reward / step
                episode_reward_list.append(mean_reward)
                episode_duration_list.append(step)
                intervene_percent.append(len(intervene_percent_per_episode[i]))

                writer.add_scalar('reward/reward_episode',          mean_reward,                          i)
                writer.add_scalar('reward/reward_episode_noshaping', np.mean(reward_e_record[i]),         i)
                writer.add_scalar('reward/duration_episode',         step,                                i)
                writer.add_scalar('percent_intervene',               len(intervene_percent_per_episode[i]), i)
                writer.add_scalar('exploration_rate',                round(exploration_rate, 4),           i)
                writer.add_scalar('loss/loss_critic',                round(np.average(c_loss), 4),        i)
                writer.add_scalar('loss/loss_actor',                 round(np.average(a_loss), 4),        i)
                break

            signal.signal(signal.SIGINT, signal_handler)

        if total_step > args.maximum_step:
            break

    print('Total training time:', time.perf_counter() - start_time)

    DRL.save_model('./TD3_based_DRL/models')

    pygame.display.quit()
    pygame.quit()

    action_drl   = previous_action[0:i]
    action_final = final_action[0:i]
    scio.savemat(
        'data{}-{}.mat'.format(args.algorithm, round(time.time())),
        mdict={
            'action_drl':              action_drl,
            'action_final':            action_final,
            'actiondisturbingdegree':  action_disturbing_degree,
            'qlist':                   qlist,
            'intervenepercent':        intervene_percent,
            'x':                       x_per_episode,
            'y':                       y_per_episode,
            'stepreward':              global_reward_list,
            'step':                    episode_duration_list,
            'reward':                  episode_reward_list,
            'r_i':                     reward_i_record,
            'r_e':                     reward_e_record,
        })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HITL-DRL Training (CARLA 0.9.15)')
    parser.add_argument('--algorithm', type=int,
                        help='0=TD3HUG, 1=TD3IARL, 2=TD3HIRL, 3=Vanilla TD3 (default: 0)',
                        default=0)
    parser.add_argument('--maximum_episode', type=float,
                        help='maximum training episodes (default: 1000)', default=1000)
    parser.add_argument('--maximum_step', type=float,
                        help='maximum training steps (default: 5e4)', default=5e4)
    parser.add_argument('--seed', type=int,
                        help='random seed (default: 2)', default=2)
    parser.add_argument('--initial_exploration_rate', type=float,
                        help='initial exploration noise (default: 0.5)', default=0.5)
    parser.add_argument('--cutoff_exploration_rate', type=float,
                        help='minimum exploration noise (default: 0.05)', default=0.05)
    parser.add_argument('--exploration_decay_rate', type=float,
                        help='exploration decay factor (default: 0.99988)', default=0.99988)
    parser.add_argument('--resume', action='store_true',
                        help='resume from checkpoint (default: False)', default=False)
    parser.add_argument('--warmup', action='store_true',
                        help='warmup before learning (default: False)', default=False)
    parser.add_argument('--warmup_threshold', type=int,
                        help='warmup steps (default: 5000)', default=5000)
    parser.add_argument('--pid_controller_guidance', action='store_true',
                        help='use PID controller as virtual human (default: False)',
                        default=False)
    parser.add_argument('--reward_shaping', type=int,
                        help='0=none, 1=intervention, 2=potential, 3=RND (default: 0)',
                        default=0)
    parser.add_argument('--device', type=str,
                        help='cuda or cpu (default: cuda)', default='cuda')
    args = parser.parse_args()

    RL_training()