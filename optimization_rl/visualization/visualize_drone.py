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

def visualize_trajectory(env, agent, max_steps=200, save_path=None):
    """Visualize the drone's trajectory in 3D"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Record trajectory
    state = env.reset()
    positions = [env.position.copy()]
    
    for _ in range(max_steps):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = agent.select_action(state_tensor, eval_mode=True)
        state, _, done, _ = env.step(action)
        positions.append(env.position.copy())
        
        if done:
            break
    
    positions = np.array(positions)
    
    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
            'b-', label='Drone Path', linewidth=2)
    
    # Plot start and end points
    ax.scatter(*positions[0], color='green', s=100, label='Start')
    ax.scatter(*positions[-1], color='red', s=100, label='End')
    
    # Plot goal
    ax.scatter(*env.goal, color='purple', s=100, label='Goal')
    
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
    ax.set_title('Drone Navigation Trajectory')
    ax.legend()
    
    # Set bounds
    ax.set_xlim([0, env.bounds[0]])
    ax.set_ylim([0, env.bounds[1]])
    ax.set_zlim([0, env.bounds[2]])
    
    if save_path:
        plt.savefig(save_path)
    
    return fig

def visualize_multiple_trajectories(env, agent, num_trajectories=5):
    """Visualize multiple trajectories to show consistency"""
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.rainbow(np.linspace(0, 1, num_trajectories))
    
    for i, color in enumerate(colors):
        state = env.reset()
        positions = [env.position.copy()]
        
        for _ in range(200):  # max steps
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = agent.select_action(state_tensor, eval_mode=True)
            state, _, done, _ = env.step(action)
            positions.append(env.position.copy())
            
            if done:
                break
        
        positions = np.array(positions)
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                color=color, label=f'Path {i+1}', linewidth=2)
    
    # Plot obstacles and goal
    for obs in env.obstacles:
        x, y, z, r = obs
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x_surf = r * np.outer(np.cos(u), np.sin(v)) + x
        y_surf = r * np.outer(np.sin(u), np.sin(v)) + y
        z_surf = r * np.outer(np.ones(np.size(u)), np.cos(v)) + z
        ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.3)
    
    ax.scatter(*env.goal, color='purple', s=100, label='Goal')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Multiple Drone Navigation Trajectories')
    ax.legend()
    
    return fig

def main():
    # Create environment
    env = DroneEnv(
        obstacles=[(5, 5, 5, 1)],
        bounds=(10, 10, 10),
        goal=(8, 8, 8)
    )
    
    # Load trained agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPO(state_dim=state_dim, action_dim=action_dim)
    
    # Load the best model
    models_dir = Path("models")
    latest_model = max(models_dir.glob("*/best_model.pth"), key=os.path.getctime)
    agent.load(latest_model)
    print(f"Loaded model from {latest_model}")
    
    # Create visualizations directory
    vis_dir = Path("visualizations")
    vis_dir.mkdir(exist_ok=True)
    
    # Generate and save visualizations
    # Single trajectory
    fig = visualize_trajectory(env, agent)
    plt.savefig(vis_dir / "single_trajectory.png")
    plt.close()
    
    # Multiple trajectories
    fig = visualize_multiple_trajectories(env, agent)
    plt.savefig(vis_dir / "multiple_trajectories.png")
    plt.close()
    
    print(f"Visualizations saved in {vis_dir}")

if __name__ == "__main__":
    main() 