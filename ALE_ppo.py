import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.atari_wrappers import *


#
name = "PPO"
gym_id = "ALE/Tetris-v5"
learning_rate = 1e-3
seed = 1
total = 10000000
torch_deterministic = True
cuda = True
track = False
capture_video = False

# values and loss calculation optimizations taken from https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
# in general, a lot of the finer details of PPO are adapted from there

nenvs = 8
steps = 128
gamma = 0.995
gae_lambda = 0.95
minibatches = 4
update_epochs = 4
clip_coef = 0.1
ent_coef = 0.01
vf_coef = 0.5
max_grad_norm = 0.5

batch_size = nenvs * steps
minibatch_size = batch_size // minibatches

def make_env(gym_id, seed, idx, capture_video, run_name):
    env = gym.make(gym_id, render_mode="rgb_array")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    if capture_video and idx == 0:
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, 4)
    
    # env.seed(seed)
    # env.action_space.seed(seed)
    # env.observation_space.seed(seed)
    
    return env

class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )
        self.actor = nn.Linear(512, envs.single_action_space.n)
        self.critic = nn.Linear(512, 1)

    def actionvalue(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
    
run_name = f"{gym_id}__{name}__{seed}__{int(time.time())}"
if track:
    import wandb

    wandb.init(
        project=name,
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )
writer = SummaryWriter(f"runs/{run_name}")


random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = torch_deterministic

device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

envs = gym.vector.SyncVectorEnv(
    [make_env(gym_id, seed + i, i, capture_video, run_name) for i in range(nenvs)]
)

agent = Agent(envs).to(device)
optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

obs = torch.zeros((steps, nenvs) + envs.single_observation_space.shape).to(device)
actions = torch.zeros((steps, nenvs) + envs.single_action_space.shape).to(device)
logprobs = torch.zeros((steps, nenvs)).to(device)
rewards = torch.zeros((steps, nenvs)).to(device)
dones = torch.zeros((steps, nenvs)).to(device)
values = torch.zeros((steps, nenvs)).to(device)

global_step = 0
start = time.time()
next_obs = torch.Tensor(envs.reset()[0]).to(device)
next_done = torch.zeros(nenvs).to(device)

for i in range(total // batch_size):
    optimizer.param_groups[0]["lr"] = (1.0 - i/ (total // batch_size))*learning_rate

    
    episode_rewards = torch.zeros(nenvs, device=device)
    all_episode_rewards = []
    episode_counts = [0] * nenvs
    episode_lengths = torch.zeros(nenvs, device=device)
    all_episode_lengths = []



    for s in range(steps):
        global_step += nenvs
        obs[s] =next_obs
        dones[s] =next_done

        with torch.no_grad():
            action, logprob, _, value = agent.actionvalue(next_obs)
            values[s] = value.flatten()
        actions[s] = action
        logprobs[s] = logprob

        next_obs, reward, done, _, _ = envs.step(action.cpu().numpy())

        rewards[s] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
        episode_rewards += reward
        episode_lengths += 1
        for i in range(nenvs):
            if done[i]:
                all_episode_rewards.append(episode_rewards[i].item())
                all_episode_lengths.append(episode_lengths[i].item())
                episode_counts[i] += 1
                episode_rewards[i] = 0
                episode_lengths[i] = 0

        if len(all_episode_rewards) > 0 and global_step % (nenvs * 100) == 0:
            avg_reward = max(all_episode_rewards[-10:])
            avg_length = sum(all_episode_lengths[-10:])/len(all_episode_lengths[-10:])
            writer.add_scalar("charts/rewards", avg_reward, global_step)
            writer.add_scalar("charts/avg_episodic_length", avg_length, global_step)

            print(f"Step: {global_step}, Rewards (Last 10 Episodes): {avg_reward:.4f},Avg Episode Length (Last 10): {avg_length:.4f}")

    with torch.no_grad():
        adv = torch.zeros_like(rewards).to(device)
        lgl = 0
        for t in reversed(range(steps)):
            if (t==steps - 1):
                terminate = 1.0 - next_done
                nextval = agent.critic(agent.network(next_obs / 255)).reshape(1, -1)
            else:
                terminate = 1.0 - dones[t+1]
                nextval=values[t+1]
            delta = rewards[t] + gamma *nextval * terminate - values[t]
            lgl = delta + gamma * gae_lambda * terminate * lgl
            adv[t] = lgl
        returns = adv + values

    batchobs = obs.reshape((-1,) + envs.single_observation_space.shape)
    batchlps = logprobs.reshape(-1)
    batchactions = actions.reshape((-1,) + envs.single_action_space.shape)
    batchadv = adv.reshape(-1)
    batchreturns = returns.reshape(-1)
    bvals = values.reshape(-1)

    b_inds = np.arange(batch_size)
    clipfracs = []
    for epoch in range(update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.actionvalue(batchobs[mb_inds], batchactions.long()[mb_inds])
            logratio = newlogprob - batchlps[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

            madv = batchadv[mb_inds]

            madv = (madv - madv.mean()) / (madv.std() + 1e-8)

            #Policy loss
            pg_loss1 = -madv * ratio
            pg_loss2 = -madv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            #Value loss
            newvalue = newvalue.view(-1)
            v_loss_unclipped = (newvalue - batchreturns[mb_inds]) ** 2
            v_clipped = bvals[mb_inds] + torch.clamp(
                newvalue - bvals[mb_inds],
                -clip_coef,
                clip_coef,
            )
            v_loss_clipped = (v_clipped - batchreturns[mb_inds]) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()


    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    print("SPS:", int(global_step / (time.time()-start)))
    writer.add_scalar("charts/SPS", int(global_step / (time.time()-start)), global_step)

envs.close()
writer.close()

torch.save(agent.state_dict(), f"ppo_tetris_{seed}.pth")
print(f"Model saved as ppo_tetris_{seed}.pth")
