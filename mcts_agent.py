import random
import copy
from tetris import Tetris

class MCTSAgent:
    def __init__(self, simulations=50, rollout_depth=10):
        self.simulations = simulations
        self.rollout_depth = rollout_depth

    def best_action(self, env):
        candidate_moves = list(env.get_next_states().keys())
        if not candidate_moves:
            return None

        best_move = None
        best_score = float('-inf')

        for move in candidate_moves:
            total_score = 0
            for _ in range(self.simulations):
                total_score += self.simulate(env, move)
            avg_score = total_score / self.simulations

            if avg_score > best_score:
                best_score = avg_score
                best_move = move

        return best_move

    def simulate(self, env, move):
        env_copy = copy.deepcopy(env)
        reward, done = env_copy.play(move[0], move[1], render=False)
        total_reward = reward

        depth = 0
        while not done and depth < self.rollout_depth:
            possible_moves = list(env_copy.get_next_states().keys())
            if not possible_moves:
                break
            random_move = random.choice(possible_moves)
            reward, done = env_copy.play(random_move[0], random_move[1], render=False)
            total_reward += reward
            depth += 1

        return total_reward
