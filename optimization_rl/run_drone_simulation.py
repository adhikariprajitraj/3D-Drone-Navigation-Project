from reinforcement_learning.training_scripts.train_rl_model import train
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import torch
from pathlib import Path

def visualize_single_path(env, agent):
    """Visualize a single path of the drone"""
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Reset and collect trajectory
    state = env.reset()
    positions = [env.position.copy()]
    rewards = []
    done = False
    
    while not done:
        # Get action and step
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = agent.select_action(state_tensor, eval_mode=True)
        state, reward, done, _ = env.step(action)
        
        # Record position and reward
        positions.append(env.position.copy())
        rewards.append(reward)
    
    # Convert positions to numpy array
    positions = np.array(positions)
    
    # Plot the path
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', label='Path')
    ax.scatter(*positions[0], color='green', s=100, label='Start')
    ax.scatter(*positions[-1], color='blue', s=100, label='End')
    ax.scatter(*env.goal, color='red', s=100, label='Goal')
    
    # Plot obstacle
    for obs in env.obstacles:
        x, y, z, r = obs
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x_surf = r * np.outer(np.cos(u), np.sin(v)) + x
        y_surf = r * np.outer(np.sin(u), np.sin(v)) + y
        z_surf = r * np.outer(np.ones(np.size(u)), np.cos(v)) + z
        ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.3)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Drone Navigation Path')
    ax.legend()
    
    # Set bounds
    ax.set_xlim([0, env.bounds[0]])
    ax.set_ylim([0, env.bounds[1]])
    ax.set_zlim([0, env.bounds[2]])
    
    return positions, rewards

def main():
    # Train the agent
    print("Training agent...")
    agent, env = train()
    
    print("\nVisualizing trained agent behavior...")
    
    # Create and show 3D trajectory plot
    positions, rewards = visualize_single_path(env, agent)
    plt.show()  # Show the 3D plot
    
    # Create and show rewards plot
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Rewards over Time')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.show()  # Show the rewards plot
    
    print("Visualization complete!")

if __name__ == "__main__":
    main() 