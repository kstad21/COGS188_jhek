import csv
from mcts_agent import MCTSAgent
from tetris import Tetris

def run_mcts():
    env = Tetris()
    agent = MCTSAgent(simulations=30, rollout_depth=15)
    
    score_history = []
    move_count = 0
    done = False

    while not done:
        best_move = agent.best_action(env)
        if best_move is None:
            print("No valid moves available. Ending game.")
            break

        reward, done = env.play(best_move[0], best_move[1], render=True)
        current_score = env.get_game_score()
        score_history.append(current_score)
        move_count += 1
        print(f"Move {move_count}: {best_move}, Reward: {reward}, Score: {current_score}")

    print("Game Over! Final Score:", env.get_game_score())
    
    with open("score_history.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Move", "Score"])
        for move, score in enumerate(score_history, start=1):
            writer.writerow([move, score])
    print("Score history saved to score_history.csv")

if __name__ == "__main__":
    run_mcts()
