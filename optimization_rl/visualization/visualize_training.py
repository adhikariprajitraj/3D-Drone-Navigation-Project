import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import torch
from pathlib import Path
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)
from optimization_rl.reinforcement_learning.ppo_agent import PPO, DroneEnv

def visualize_live_trajectory(env, agent):
    """Visualize the drone's trajectory in real-time"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize trajectory storage
    positions = []
    rewards = []
    
    # Reset environment
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Get current position
        positions.append(env.position.copy())
        
        # Get action from agent
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = agent.select_action(state_tensor, eval_mode=True)
        
        # Take step
        state, reward, done, _ = env.step(action)
        total_reward += reward
        rewards.append(reward)
        
        # Clear previous plot
        ax.clear()
        
        # Plot trajectory
        positions_array = np.array(positions)
        ax.plot(positions_array[:, 0], positions_array[:, 1], positions_array[:, 2], 'b-', label='Trajectory')
        
        # Plot current position
        ax.scatter(*positions[-1], color='blue', s=100, label='Current Position')
        
        # Plot start
        ax.scatter(*positions[0], color='green', s=100, label='Start')
        
        # Plot goal
        ax.scatter(*env.goal, color='red', s=100, label='Goal')
        
        # Plot obstacles
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
        ax.set_title(f'Drone Navigation (Reward: {total_reward:.2f})')
        
        # Set consistent bounds
        ax.set_xlim([0, env.bounds[0]])
        ax.set_ylim([0, env.bounds[1]])
        ax.set_zlim([0, env.bounds[2]])
        
        ax.legend()
        
        plt.pause(0.01)  # Small pause to update display
    
    plt.show()
    return positions, rewards

def main():
    # Create environment
    env = DroneEnv(
        obstacles=[(5, 5, 5, 1)],
        bounds=(10, 10, 10),
        goal=(8, 8, 8)
    )
    
    # Initialize agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPO(state_dim=state_dim, action_dim=action_dim)
    
    # Load the best model if it exists
    models_dir = Path("models")
    if models_dir.exists():
        try:
            latest_model = max(models_dir.glob("*/best_model.pth"), key=os.path.getctime)
            agent.load(latest_model)
            print(f"Loaded model from {latest_model}")
        except ValueError:
            print("No trained model found. Using untrained agent.")
    
    # Visualize trajectory
    print("Starting visualization...")
    positions, rewards = visualize_live_trajectory(env, agent)
    
    # Plot reward history
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Rewards over Time')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.show()
    
    # Print summary
    print(f"Total steps: {len(positions)}")
    print(f"Final position: {positions[-1]}")
    print(f"Goal position: {env.goal}")
    print(f"Total reward: {sum(rewards):.2f}")

if __name__ == "__main__":
    main() 