import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import torch

def visualize_trajectory(env, agent, max_steps=200):
    """Create a visualization of the drone's trajectory"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Record trajectory
    state = env.reset()
    positions = [env.position.copy()]
    
    for _ in range(max_steps):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, _ = agent.get_action(state_tensor)
        action = action.numpy().squeeze()
        
        state, _, done, _ = env.step(action)
        positions.append(env.position.copy())
        
        if done:
            break
    
    positions = np.array(positions)
    
    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', label='Trajectory')
    
    # Plot obstacles
    for obs in env.obstacles:
        x, y, z, r = obs
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x_surf = r * np.outer(np.cos(u), np.sin(v)) + x
        y_surf = r * np.outer(np.sin(u), np.sin(v)) + y
        z_surf = r * np.outer(np.ones(np.size(u)), np.cos(v)) + z
        ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.3)
    
    # Plot start and goal
    ax.scatter(*env.position, color='green', s=100, label='Start')
    ax.scatter(*env.goal, color='red', s=100, label='Goal')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    return fig 