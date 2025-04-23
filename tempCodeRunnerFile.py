import argparse
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

from utils import gym_wrappers
from utils import model

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.5

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02
TRAIN_FREQ = 4


Transition = collections.namedtuple(
    'Transition', ['state', 'action', 'reward', 'done', 'new_state']
)


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[i] for i in indices])
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.uint8),
            np.array(next_states),
        )


class Agent:
    def __init__(self, env, exp_buffer, device):
        self.env = env
        self.buffer = exp_buffer
        self.device = device
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def step_env(self, net, epsilon, render=False):
        if np.random.rand() < epsilon:
            action = self.env.action_space.sample()
        else:
            st = torch.tensor(self.state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = net(st)
            action = int(q_vals.max(1)[1].item())

        new_state, reward, done, _ = self.env.step(action)

        if render:
            self.env.render()  # 👈 Show the Pong game

        self.total_reward += reward
        self.buffer.append(Transition(self.state, action, reward, done, new_state))
        self.state = new_state

        if done:
            dr = self.total_reward
            self._reset()
            return dr
        return None



def compute_loss(batch, net, tgt_net, device):
    states, actions, rewards, dones, next_states = batch

    s_v = torch.tensor(states, dtype=torch.float32, device=device)
    ns_v = torch.tensor(next_states, dtype=torch.float32, device=device)
    acts_v = torch.tensor(actions, dtype=torch.int64, device=device)
    rews_v = torch.tensor(rewards, dtype=torch.float32, device=device)
    done_mask = torch.tensor(dones, dtype=torch.bool, device=device)

    q_vals = net(s_v).gather(1, acts_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_q = tgt_net(ns_v).max(1)[0]
        next_q[done_mask] = 0.0
    expected_q = rews_v + GAMMA * next_q

    return nn.MSELoss()(q_vals, expected_q)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cuda", action="store_true", help="Enable CUDA")
    p.add_argument("--env", default=DEFAULT_ENV_NAME)
    p.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND)
    args = p.parse_args()

    use_cuda = args.cuda and torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    print("Using device:", device)
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    env     = gym_wrappers.make_env(args.env, render_mode="human")
    net     = model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = model.DQN(env.observation_space.shape, env.action_space.n).to(device)

    buf   = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buf, device)
    writer= SummaryWriter(comment="-" + args.env)
    opt   = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    eps_history   = []
    speed_history = []
    mean_history  = []
    frame_idx     = 0
    best_mean     = None
    start_time    = time.time()

    try:
        while True:
            frame_idx += 1
            eps = max(EPSILON_FINAL,
                      EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

            reward = agent.step_env(net, eps, render=True)
            if reward is not None:
                total_rewards.append(reward)
                elapsed = time.time() - start_time
                speed   = frame_idx / elapsed
                mean_100= np.mean(total_rewards[-100:])

                # log to console & TensorBoard
                print(f"{frame_idx}: games {len(total_rewards)}, "
                      f"mean {mean_100:.3f}, eps {eps:.2f}, {speed:.2f} f/s")
                writer.add_scalar("mean_reward", mean_100, frame_idx)
                writer.add_scalar("speed",       speed,   frame_idx)
                writer.add_scalar("epsilon",     eps,     frame_idx)

                # store for post-mortem plotting
                eps_history.append(eps)
                speed_history.append(speed)
                mean_history.append(mean_100)

                if best_mean is None or best_mean < mean_100:
                    torch.save(net.state_dict(), args.env + "-best.dat")
                    best_mean = mean_100
                    print("New best mean:", best_mean)

                if mean_100 > args.reward:
                    print("Solved in", frame_idx, "frames!")
                    break

            if len(buf) >= REPLAY_START_SIZE:
                if frame_idx % SYNC_TARGET_FRAMES == 0:
                    tgt_net.load_state_dict(net.state_dict())

                if frame_idx % TRAIN_FREQ == 0:
                    batch = buf.sample(BATCH_SIZE)
                    loss  = compute_loss(batch, net, tgt_net, device)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()


    except KeyboardInterrupt:
        print("\n⏹  Training interrupted by user — saving logs and metrics…")

    finally:
        # flush TensorBoard logs
        writer.close()

        # save raw metrics
        np.savez("metrics.npz",
                 total_rewards=np.array(total_rewards),
                 frames=np.arange(len(mean_history)) * SYNC_TARGET_FRAMES,
                 mean_reward=np.array(mean_history),
                 epsilon=np.array(eps_history),
                 speed=np.array(speed_history))
        print("• metrics.npz written")

        # emit quick PNGs
        import matplotlib.pyplot as plt

        plt.plot(mean_history)
        plt.title("Mean Reward (100)"); plt.xlabel("Update #"); plt.ylabel("Mean Reward")
        plt.savefig("mean_reward.png"); plt.close()

        plt.plot(eps_history)
        plt.title("Epsilon Decay"); plt.xlabel("Update #"); plt.ylabel("Epsilon")
        plt.savefig("epsilon.png"); plt.close()

        plt.plot(speed_history)
        plt.title("Training Speed"); plt.xlabel("Update #"); plt.ylabel("Frames/s")
        plt.savefig("speed.png"); plt.close()

        print("• mean_reward.png, epsilon.png, speed.png written")

