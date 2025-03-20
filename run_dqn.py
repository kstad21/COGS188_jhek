import torch
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm
from statistics import mean
from torch.utils.tensorboard import SummaryWriter

from dqn_agent import DQNAgent
from tetris import Tetris


class Logger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def record(self, step, **metrics):
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)


def train_dqn():
    env = Tetris()
    config = {
        "episodes": 3000,
        "max_steps": None,
        "epsilon_stop_episode": 2000,
        "mem_size": 1000,
        "discount": 0.6,
        "batch_size": 128,
        "epochs": 1,
        "render_every": 50,
        "render_delay": None,
        "log_every": 50,
        "replay_start_size": 1000,
        "train_every": 1,
        "n_neurons": [32, 32, 32],
        "activations": ['relu', 'relu', 'relu', 'linear'],
        "save_best_model": True,
    }

    agent = DQNAgent(
        env.get_state_size(),
        n_neurons=config["n_neurons"],
        activations=config["activations"],
        epsilon_stop_episode=config["epsilon_stop_episode"],
        mem_size=config["mem_size"],
        discount=config["discount"],
        replay_start_size=config["replay_start_size"]
    )

    log_path = f'logs/tetris-nn={config["n_neurons"]}-mem={config["mem_size"]}-bs={config["batch_size"]}-e={config["epochs"]}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    logger = Logger(log_path)

    scores = []
    highest_score = 0

    for episode in tqdm(range(config["episodes"]), desc="Training Progress"):
        state = env.reset()
        game_over = False
        steps = 0
        render = episode % config["render_every"] == 0 if config["render_every"] else False

        while not game_over and (not config["max_steps"] or steps < config["max_steps"]):
            next_states = {tuple(v): k for k, v in env.get_next_states().items()}
            best_state = agent.best_state(next_states.keys())
            best_action = next_states[best_state]

            reward, game_over = env.play(best_action[0], best_action[1], render=render, render_delay=config["render_delay"])
            agent.add_to_memory(state, best_state, reward, game_over)
            state = best_state
            steps += 1

        scores.append(env.get_game_score())

        if episode % config["train_every"] == 0:
            agent.train(batch_size=config["batch_size"], epochs=config["epochs"])

        if config["log_every"] and episode % config["log_every"] == 0:
            avg_score = mean(scores[-config["log_every"]:])
            min_score = min(scores[-config["log_every"]:])
            max_score = max(scores[-config["log_every"]:])
            logger.record(episode, avg_score=avg_score, min_score=min_score, max_score=max_score)

        if config["save_best_model"] and env.get_game_score() > highest_score:
            print(f"New best model! Score: {env.get_game_score()} at Episode {episode}")
            highest_score = env.get_game_score()
            agent.save_model("best.pth")


if __name__ == "__main__":
    train_dqn()
