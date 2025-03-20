import torch
from datetime import datetime
from tqdm import tqdm
from statistics import mean
from torch.utils.tensorboard import SummaryWriter

from ppo_agent import PPOAgent
from tetris import Tetris

class Logger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
    def record(self, step, **metrics):
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

config = {
    "episodes": 3000,           
    "max_steps": None,          
    "update_epochs": 4,         
    "render_every": 50,         
    "log_every": 50,            
    "save_best_model": True,     
    
    "gamma": 0.95,              
    "clip_epsilon": 0.2,         
    "lr": 1e-3,                  
    
    "dropout_rate": 0.2,        
    "n_neurons": [128, 128, 64, 32]  
}

action_space = []
for x in range(-3, 11): 
    for rotation in [0, 90, 180, 270]:
        action_space.append((x, rotation))
action_dim = len(action_space)

agent = PPOAgent(
    state_size=Tetris().get_state_size(), 
    action_dim=action_dim,
    n_neurons=config["n_neurons"],         
    activations=['relu'] * len(config["n_neurons"]),
    gamma=config["gamma"],                  
    clip_epsilon=config["clip_epsilon"],    
    update_epochs=config["update_epochs"], 
    lr=config["lr"],                        
    dropout_rate=config["dropout_rate"]   
)

log_path = f'logs/tetris-ppo-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
logger = Logger(log_path)



def train_ppo():
    scores = []
    highest_score = 0
    env = Tetris()
    for episode in tqdm(range(config["episodes"]), desc="Training PPO"):
        state = env.reset()
        game_over = False
        episode_reward = 0
        steps = 0
        render = (episode % config["render_every"] == 0) if config.get("render_every") else False

        while not game_over and (config.get("max_steps") is None or steps < config.get("max_steps")):
            next_states_dict = env.get_next_states()
            valid_action_indices = [idx for idx, action in enumerate(action_space) if action in next_states_dict]
            if not valid_action_indices:
                break  

            action_idx, log_prob, value, probs = agent.select_action(state, valid_action_indices)
            chosen_move = action_space[action_idx]
            candidate_state = next_states_dict.get(chosen_move, state)

            reward, game_over = env.play(chosen_move[0], chosen_move[1], render=render)
            agent.store_transition((state, valid_action_indices, action_idx, log_prob, reward, game_over))
            state = candidate_state
            episode_reward += reward
            steps += 1

        scores.append(env.get_game_score())
        agent.finish_episode()

        if config.get("log_every") and episode % config["log_every"] == 0:
            avg_score = mean(scores[-config["log_every"]:])
            min_score = min(scores[-config["log_every"]:])
            max_score = max(scores[-config["log_every"]:])
            logger.record(episode, avg_score=avg_score, min_score=min_score, max_score=max_score)

        if config.get("save_best_model") and env.get_game_score() > highest_score:
            print(f"New best model! Score: {env.get_game_score()} at Episode {episode}")
            highest_score = env.get_game_score()
            agent.save_model("ppo_best.pth")

if __name__ == "__main__":
    train_ppo()
