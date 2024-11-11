import os
import sys
import torch

# Add the optimization_rl and environment directories to the system path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'optimization_rl'))
sys.path.append(os.path.join(project_root, 'environment'))

from optimization_rl.reinforcement_learning.training_scripts.train_rl_model import train, visualize_learned_policy

def main():
    # Train the agent
    agent, env, optimal_path, start, goal = train()
    
    if agent is not None:
        # Load the best model and visualize final performance
        agent.load_state_dict(torch.load("models/best_policy.pth"))
        # Use the visualization function from train_rl_model.py instead
        visualize_learned_policy(agent, env, optimal_path, start, goal)

if __name__ == "__main__":
    main()
