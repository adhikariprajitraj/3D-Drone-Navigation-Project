import os
import numpy as np
from datetime import datetime
from pathlib import Path
import torch
import wandb
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import random

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(project_root)
from optimization_rl.reinforcement_learning.ppo_agent import PPO
from environment.environment_setup import DynamicEnvironment
from optimization_rl.path_planning.rrt import RRT

class DynamicEnvironmentVisualizer:
    def __init__(self, bounds, static_obstacles, dynamic_obstacles, optimal_path=None):
        self.bounds = bounds
        self.static_obstacles = static_obstacles
        self.dynamic_obstacles = dynamic_obstacles
        self.optimal_path = optimal_path
        
        # Setup visualization
        self.fig = plt.figure(figsize=(15, 5))
        
        # Create three subplots: RRT path, Current State, Learning Progress
        self.ax1 = self.fig.add_subplot(131, projection='3d')  # RRT path
        self.ax2 = self.fig.add_subplot(132, projection='3d')  # Current State
        self.ax3 = self.fig.add_subplot(133)  # Learning Progress
        
        self.reward_history = []
        self.episode_steps = []
        
    def update_training_plot(self, episode, reward, steps):
        self.reward_history.append(reward)
        self.episode_steps.append(steps)
        
        self.ax3.clear()
        self.ax3.plot(self.reward_history, label='Reward')
        self.ax3.set_xlabel('Episode')
        self.ax3.set_ylabel('Total Reward')
        self.ax3.set_title('Training Progress')
        self.ax3.legend()
        plt.pause(0.01)

    def plot_environment(self, ax, drone_pos=None, highlight_dynamic=False):
        ax.clear()
        
        # Plot static obstacles
        for obs in self.static_obstacles:
            x, y, z, r = obs
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x_surf = r * np.outer(np.cos(u), np.sin(v)) + x
            y_surf = r * np.outer(np.sin(u), np.sin(v)) + y
            z_surf = r * np.outer(np.ones(np.size(u)), np.cos(v)) + z
            ax.plot_surface(x_surf, y_surf, z_surf, color='gray', alpha=0.3)
        
        # Plot dynamic obstacles
        for obs in self.dynamic_obstacles:
            x, y, z = obs['position']
            r = obs['radius']
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x_surf = r * np.outer(np.cos(u), np.sin(v)) + x
            y_surf = r * np.outer(np.sin(u), np.sin(v)) + y
            z_surf = r * np.outer(np.ones(np.size(u)), np.cos(v)) + z
            color = 'red' if highlight_dynamic else 'blue'
            alpha = 0.7 if highlight_dynamic else 0.3
            ax.plot_surface(x_surf, y_surf, z_surf, color=color, alpha=alpha)
        
        # Plot optimal path if available
        if self.optimal_path is not None:
            path = np.array(self.optimal_path)
            ax.plot(path[:, 0], path[:, 1], path[:, 2], 'g--', label='RRT Path')
        
        # Plot drone position if available
        if drone_pos is not None:
            ax.scatter(*drone_pos, color='green', s=100, label='Drone')
        
        # Set labels and limits
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(0, self.bounds[0])
        ax.set_ylim(0, self.bounds[1])
        ax.set_zlim(0, self.bounds[2])
        ax.legend()

    def update_visualization(self, drone_pos):
        # Update current state plot
        self.plot_environment(self.ax2, drone_pos, highlight_dynamic=True)
        plt.pause(0.01)

def visualize_initial_setup(env, optimal_path, start, goal):
    """Visualize initial RRT path and environment setup"""
    viz = DynamicEnvironmentVisualizer(env.bounds, env.static_obstacles, 
                                     env.dynamic_obstacles, optimal_path)
    
    # Plot initial setup
    viz.plot_environment(viz.ax1)
    viz.ax1.scatter(*start, color='green', s=100, label='Start')
    viz.ax1.scatter(*goal, color='red', s=100, label='Goal')
    viz.ax1.set_title('Initial RRT Path')
    
    plt.show(block=False)
    return viz

def setup_environment():
    """Setup the training environment and generate RRT path"""
    bounds = (10, 10, 10)
    start = (1, 1, 1)
    goal = (8, 8, 8)
    
    # Generate static obstacles
    static_obstacles = []
    for _ in range(10):
        x = random.uniform(2, 8)
        y = random.uniform(2, 8)
        z = random.uniform(2, 8)
        radius = random.uniform(0.5, 1.0)
        static_obstacles.append((x, y, z, radius))
    
    # Add dynamic obstacles
    dynamic_obstacles = []
    for _ in range(4):  # Add 4 dynamic obstacles
        dynamic_obstacles.append({
            'position': [random.uniform(2, 8), random.uniform(2, 8), random.uniform(2, 8)],
            'velocity': [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)],
            'radius': random.uniform(0.3, 0.7)
        })
    
    # Initialize environment with both static and dynamic obstacles
    env = DynamicEnvironment(bounds, static_obstacles, dynamic_obstacles)
    
    # Get RRT path
    rrt = RRT(start, goal, static_obstacles, bounds)
    optimal_path = rrt.plan()
    
    if optimal_path is None:
        print("RRT couldn't find a path. Adjusting environment...")
        return None, None, None, None, None, None
    
    return env, optimal_path, start, goal, bounds, static_obstacles

def compute_path_following_reward(current_pos, optimal_path, closest_point_idx):
    """Compute reward for following the RRT path"""
    # Find the closest point on the path ahead of the current closest point
    look_ahead = 3  # Number of points to look ahead
    target_idx = min(closest_point_idx + look_ahead, len(optimal_path) - 1)
    target_point = optimal_path[target_idx]
    
    # Distance to target point
    distance = np.linalg.norm(np.array(current_pos) - np.array(target_point))
    path_reward = -distance * 2.0  # Increased weight for path following
    
    # Add bonus for staying close to path
    min_path_distance = float('inf')
    for path_point in optimal_path[closest_point_idx:target_idx+1]:
        dist = np.linalg.norm(np.array(current_pos) - np.array(path_point))
        min_path_distance = min(min_path_distance, dist)
    
    path_reward -= min_path_distance  # Additional penalty for deviating from path
    
    return path_reward, target_idx

def get_closest_obstacle(current_pos, dynamic_obstacles, detection_radius=2.0):
    """Find the closest dynamic obstacle within detection radius"""
    closest_dist = float('inf')
    closest_obstacle = None
    
    for obs in dynamic_obstacles:
        dist = np.linalg.norm(np.array(current_pos) - np.array(obs['position']))
        if dist < closest_dist and dist < detection_radius:
            closest_dist = dist
            closest_obstacle = obs
            
    return closest_obstacle, closest_dist

def compute_obstacle_avoidance_vector(current_pos, obstacle):
    """Compute avoidance vector away from obstacle"""
    direction = np.array(current_pos) - np.array(obstacle['position'])
    distance = np.linalg.norm(direction)
    if distance < 0.1:  # Prevent division by zero
        distance = 0.1
    normalized_direction = direction / distance
    
    # Consider obstacle velocity for better avoidance
    velocity = np.array(obstacle['velocity'])
    avoidance_vector = normalized_direction + 0.5 * velocity
    
    return avoidance_vector

def train():
    """Train the RL agent"""
    # Initialize wandb at the start of training
    wandb.init(
        project="drone-path-planning",
        config={
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "clip_range": 0.2,
            "value_coef": 0.5,
            "entropy_coef": 0.01,
            "num_episodes": 10,
            "max_steps": 500
        }
    )

    # Setup environment
    env, optimal_path, start, goal, bounds, static_obstacles = setup_environment()
    if env is None:
        return None, None, None, None, None
    
    # Initialize visualizer
    viz = visualize_initial_setup(env, optimal_path, start, goal)
    
    # Initialize RL agent
    state_representation = env.get_state_representation(list(start))
    state_representation_size = len(state_representation)
    path_representation_size = len(optimal_path) * 3  # 3 coordinates per point
    total_state_dim = state_representation_size + path_representation_size

    # Ensure consistent state dimension
    max_path_points = 15  # Set a fixed maximum number of path points
    fixed_path_dim = max_path_points * 3  # 3 coordinates per point
    state_dim = state_representation_size + fixed_path_dim

    agent = PPO(state_dim=state_dim, 
               action_dim=4,
               lr=0.0003,
               gamma=0.99,
               clip_range=0.2,
               value_coef=0.5,
               entropy_coef=0.01)
    
    # Training loop
    num_episodes = 20
    max_steps = 500
    best_reward = float('-inf')
    
    for episode in range(num_episodes):
        state = list(start)
        env.reset()
        total_reward = 0
        closest_path_idx = 0
        current_path = optimal_path
        
        for step in range(max_steps):
            # Update visualization
            if episode % 10 == 0 and step % 5 == 0:
                viz.update_visualization(state)
                viz.plot_environment(viz.ax1)
                viz.ax1.plot(np.array(current_path)[:, 0], 
                            np.array(current_path)[:, 1], 
                            np.array(current_path)[:, 2], 'g--', label='Current Path')
                plt.pause(0.01)
            
            # Get closest dynamic obstacle
            closest_obstacle, obstacle_distance = get_closest_obstacle(state, env.dynamic_obstacles)
            
            # If obstacle is too close, recalculate RRT path
            if closest_obstacle and obstacle_distance < 1.5:
                # Calculate new RRT path from current position to goal
                rrt = RRT(state, goal, static_obstacles, bounds)
                new_path = rrt.plan()
                if new_path is not None:
                    current_path = new_path
                    closest_path_idx = 0  # Reset path index for new path
            
            # Get target point on path
            target_point = current_path[min(closest_path_idx + 1, len(current_path) - 1)]
            direction_to_target = np.array(target_point) - np.array(state)
            distance_to_target = np.linalg.norm(direction_to_target)
            
            if distance_to_target < 0.5:  # If close to current target, move to next point
                closest_path_idx = min(closest_path_idx + 1, len(current_path) - 1)
            
            # Determine action based on situation
            state_representation = env.get_state_representation(state)
            path_representation = np.array(current_path).flatten()
            
            # Ensure path_representation has consistent size by padding or truncating
            max_path_length = (state_dim - len(state_representation)) // 3
            if len(current_path) > max_path_length:
                path_representation = path_representation[:max_path_length*3]
            else:
                # Pad with zeros if path is shorter
                padding = np.zeros(max_path_length*3 - len(path_representation))
                path_representation = np.concatenate([path_representation, padding])
            
            state_full = np.concatenate([state_representation, path_representation])
            
            state_tensor = torch.FloatTensor(state_full).unsqueeze(0)
            
            if closest_obstacle and obstacle_distance < 1.0:
                # Obstacle avoidance mode
                avoidance_vector = compute_obstacle_avoidance_vector(state, closest_obstacle)
                # Combine avoidance with path following
                normalized_direction = direction_to_target / (distance_to_target + 1e-6)
                combined_direction = 0.7 * avoidance_vector + 0.3 * normalized_direction
                combined_direction = combined_direction / (np.linalg.norm(combined_direction) + 1e-6)
                action = torch.FloatTensor(np.append(combined_direction, 0.0))  # Add zero for yaw
            else:
                # Path following mode
                normalized_direction = direction_to_target / (distance_to_target + 1e-6)
                action = torch.FloatTensor(np.append(normalized_direction, 0.0))  # Add zero for yaw
            
            # Execute action
            next_state = np.array(state) + action.numpy()[:3] * 0.5
            
            # Compute rewards
            path_following_reward = -distance_to_target
            obstacle_penalty = -10.0 if closest_obstacle and obstacle_distance < 0.5 else 0.0
            goal_reward, done = env.compute_reward(next_state, goal)
            
            total_step_reward = (
                1.0 * path_following_reward +
                1.0 * obstacle_penalty +
                2.0 * goal_reward
            )
            
            # Store transition
            action = action.view(1, -1)
            agent.store_transition(
                state_tensor,
                action,
                torch.FloatTensor([total_step_reward]),
                next_state,
                torch.tensor([done])
            )
            
            state = next_state.tolist()
            total_reward += total_step_reward
            
            if done:
                break
            
            # Update environment
            env.update()
        
        # Update visualization with training progress
        viz.update_training_plot(episode, total_reward, step + 1)
        
        # Update agent
        if episode % 5 == 0:
            # Process transitions only if there are any stored
            transitions = agent.process_transitions()
            if transitions:  # Check if we have any transitions to process
                states, actions, rewards, next_states, dones = transitions
                
                # Add debug prints
                print(f"Buffer sizes - States: {len(states)}, Rewards: {len(rewards)}, Dones: {len(dones)}")
                
                returns = agent.compute_returns(rewards, dones)
                advantages = agent.compute_advantages(returns, states)
                
                old_log_probs = agent.get_log_probs(states, actions)
                
                # Add size checks before update
                if len(returns) > 0 and len(states) > 0:
                    agent.update(states, actions, old_log_probs, returns, advantages)
                else:
                    print("Warning: Empty batch, skipping update")
            else:
                print("Warning: No transitions to process, skipping update")
        
        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(agent.state_dict(), "models/best_policy.pth")
        
        print(f"Episode {episode} | Reward: {total_reward:.2f} | Steps: {step+1}")
        
        # Log metrics after each episode
        wandb.log({
            "episode": episode,
            "total_reward": total_reward,
            "episode_length": step + 1,
            "distance_to_goal": np.linalg.norm(np.array(state) - np.array(goal)),
            "best_reward": best_reward
        })

        # During training updates, log additional metrics
        if episode % 5 == 0 and transitions:
            wandb.log({
                "mean_advantage": advantages.mean().item(),
                "mean_return": returns.mean().item()
            })
            # Only log losses if they were returned by the update method
            if hasattr(agent, 'latest_policy_loss') and hasattr(agent, 'latest_value_loss'):
                wandb.log({
                    "policy_loss": agent.latest_policy_loss,
                    "value_loss": agent.latest_value_loss
                })

    # Close wandb run
    wandb.finish()

    return agent, env, optimal_path, start, goal

def visualize_learned_policy(agent, env, optimal_path, start, goal, scenario_name="default"):
    """Visualize the learned policy execution"""
    print("Starting visualization...")
    
    # Create a single figure with one 3D subplot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Add text annotation for progress
    text_annotation = ax.text2D(0.02, 0.98, '', transform=ax.transAxes, 
                               verticalalignment='top')
    
    state = list(start)
    done = False
    trajectory = [state]
    max_steps = 500
    step = 0
    closest_idx = 0
    
    # Store control values during trajectory
    control_history = {
        'roll': [],
        'pitch': [],
        'yaw': [],
        'throttle': []
    }
    
    while not done and step < max_steps:
        step += 1
        
        # Get state representation
        state_representation = env.get_state_representation(state)
        path_representation = np.array(optimal_path).flatten()
        
        # Ensure consistent size
        max_path_points = 15
        max_path_length = (agent.actor_critic.features[0].in_features - len(state_representation)) // 3
        
        if len(optimal_path) > max_path_length:
            path_representation = path_representation[:max_path_length*3]
        else:
            padding = np.zeros(max_path_length*3 - len(path_representation))
            path_representation = np.concatenate([path_representation, padding])
        
        state_full = np.concatenate([state_representation, path_representation])
        state_tensor = torch.FloatTensor(state_full).unsqueeze(0)
        
        # Get action from policy
        with torch.no_grad():
            action = agent.select_action(state_tensor, eval_mode=True)
            if isinstance(action, torch.Tensor):
                action = action.numpy()
        
        # Normalize action vector
        action_direction = action[:3]
        action_norm = np.linalg.norm(action_direction)
        if action_norm > 1e-6:
            action_direction = action_direction / action_norm
        
        # Find next target point on optimal path
        current_pos = np.array(state)
        
        # Find closest point on optimal path
        distances = [np.linalg.norm(np.array(p) - current_pos) for p in optimal_path]
        closest_idx = min(range(len(distances)), key=distances.__getitem__)
        
        # Look ahead on the path
        look_ahead = 3
        target_idx = min(closest_idx + look_ahead, len(optimal_path) - 1)
        target_point = optimal_path[target_idx]
        
        # Calculate direction to target and goal
        to_target = np.array(target_point) - current_pos
        to_goal = np.array(goal) - current_pos
        
        dist_to_target = np.linalg.norm(to_target)
        dist_to_goal = np.linalg.norm(to_goal)
        
        # Normalize directions
        if dist_to_target > 1e-6:
            target_direction = to_target / dist_to_target
        else:
            target_direction = np.zeros(3)
            
        if dist_to_goal > 1e-6:
            goal_direction = to_goal / dist_to_goal
        else:
            goal_direction = np.zeros(3)
        
        # Combine directions: policy, target point, and goal
        combined_direction = (0.4 * action_direction + 
                            0.4 * target_direction +
                            0.2 * goal_direction)
        
        # Normalize combined direction
        combined_norm = np.linalg.norm(combined_direction)
        if combined_norm > 1e-6:
            combined_direction = combined_direction / combined_norm
        
        # Take step with adaptive step size
        step_size = min(0.2, dist_to_goal / 10)  # Smaller steps near goal
        next_state = current_pos + combined_direction * step_size
        
        # Enforce bounds with margin
        margin = 0.1
        for i in range(3):
            next_state[i] = max(margin, min(next_state[i], env.bounds[i] - margin))
        
        state = next_state.tolist()
        trajectory.append(state)
        
        # Update visualization
        ax.clear()
        
        # Plot static obstacles
        for obs in env.static_obstacles:
            x, y, z, r = obs
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x_surf = r * np.outer(np.cos(u), np.sin(v)) + x
            y_surf = r * np.outer(np.sin(u), np.sin(v)) + y
            z_surf = r * np.outer(np.ones(np.size(u)), np.cos(v)) + z
            ax.plot_surface(x_surf, y_surf, z_surf, color='gray', alpha=0.3)
        
        # Plot dynamic obstacles
        for obs in env.dynamic_obstacles:
            x, y, z = obs['position']
            r = obs['radius']
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x_surf = r * np.outer(np.cos(u), np.sin(v)) + x
            y_surf = r * np.outer(np.sin(u), np.sin(v)) + y
            z_surf = r * np.outer(np.ones(np.size(u)), np.cos(v)) + z
            ax.plot_surface(x_surf, y_surf, z_surf, color='blue', alpha=0.3)
        
        # Plot optimal path
        path = np.array(optimal_path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], 'g--', label='RRT Path')
        
        # Plot current trajectory
        trajectory_array = np.array(trajectory)
        ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], trajectory_array[:, 2], 
               'b-', linewidth=2, label='Learned Path')
        
        # Plot start and goal
        ax.scatter(*start, color='green', s=100, label='Start')
        ax.scatter(*goal, color='red', s=100, label='Goal')
        
        # Plot current position
        ax.scatter(*state, color='blue', s=100, label='Current Position')
        
        # Set labels and limits
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(0, env.bounds[0])
        ax.set_ylim(0, env.bounds[1])
        ax.set_zlim(0, env.bounds[2])
        ax.legend()
        
        # Update progress text
        dist_to_goal = np.linalg.norm(np.array(state) - np.array(goal))
        status_text = f'Step: {step}\nPosition: [{state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}]\nDistance to goal: {dist_to_goal:.2f}'
        ax.text(env.bounds[0] * 0.05, env.bounds[1] * 0.95, env.bounds[2] * 0.95, 
                status_text, fontsize=8)
        
        plt.pause(0.01)
        
        # Print progress
        if step % 10 == 0:
            print(f"Step {step}, Position: {[f'{x:.2f}' for x in state]}, Distance to goal: {dist_to_goal:.2f}")
        
        # Check if goal is reached
        if dist_to_goal < 0.01:
            state = goal  # Set final position exactly at goal
            trajectory.append(state)
            done = True
            print("Goal reached!")
        
        # Store control values (from the action)
        control_history['roll'].append(action[0])
        control_history['pitch'].append(action[1])
        control_history['yaw'].append(action[2])
        control_history['throttle'].append(action[3] if len(action) > 3 else 0)
    
    # After trajectory completion, create control value plots
    fig_controls, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig_controls.suptitle('Control Values Over Time')
    
    for idx, (control, values) in enumerate(control_history.items()):
        row = idx // 2
        col = idx % 2
        axs[row, col].plot(values, label=control)
        axs[row, col].set_title(f'{control.capitalize()} vs Time')
        axs[row, col].set_xlabel('Time Step')
        axs[row, col].set_ylabel(f'{control.capitalize()} Value')
        axs[row, col].grid(True)
        axs[row, col].legend()
    
    plt.tight_layout()
    plt.savefig(f'control_values_{scenario_name}.png')
    print(f"Control values saved to control_values_{scenario_name}.png")
    plt.close(fig_controls)

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Train the agent
    agent, env, optimal_path, start, goal = train()
    
    if agent is not None:
        # Load the best model and visualize final performance
        agent.load_state_dict(torch.load("models/best_policy.pth"))
        visualize_learned_policy(agent, env, optimal_path, start, goal)