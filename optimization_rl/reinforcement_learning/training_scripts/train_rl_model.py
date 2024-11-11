import os
import numpy as np
from datetime import datetime
from pathlib import Path
import torch
import wandb
import sys
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(project_root)
from optimization_rl.reinforcement_learning.ppo_agent import PPO, DroneEnv
from optimization_rl.path_planning.rrt import RRT

def train():
    # Initialize environment
    env = DroneEnv(
        obstacles=[(5, 5, 5, 1),
                  (3, 7, 2, 0.5),
                  (8, 2, 4, 0.7)],
        bounds=(10, 10, 10),
        goal=(8, 8, 8)
    )
    
    # Initialize RRT planner for guidance
    rrt = RRT(
        start=(0, 0, 0),
        goal=env.goal,
        obstacles=env.obstacles,
        bounds=env.bounds,
        step_size=0.5,
        goal_sample_rate=0.2
    )
    
    # Get initial path from RRT
    reference_path = rrt.plan()
    
    # Get state and action dimensions from environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Initialize agent with tuned parameters
    agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        clip_range=0.2,
        value_coef=0.5,
        entropy_coef=0.01
    )
    
    # Training parameters
    num_episodes = 1000
    max_steps_per_episode = 500  # Increased step limit
    num_steps_per_update = 2048  # Increased batch size
    steps_since_update = 0
    
    # Create save directory
    save_dir = Path("models") / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project="drone-navigation",
        config={
            "algorithm": "PPO",
            "num_episodes": num_episodes,
            "max_steps": max_steps_per_episode,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "clip_range": 0.2,
            "value_coef": 0.5,
            "entropy_coef": 0.01,
            "num_steps_per_update": num_steps_per_update,
            "state_dim": state_dim,
            "action_dim": action_dim
        }
    )
    
    best_reward = float('-inf')
    episode_rewards = []
    
    # Setup visualization
    plt.ion()
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        positions = [env.position.copy()]
        done = False
        steps = 0
        
        # Find closest point in reference path for initial guidance
        current_path_idx = 0
        
        while not done and steps < max_steps_per_episode:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Use RRT guidance in early training
            if episode < num_episodes * 0.3 and reference_path:  # First 30% of episodes
                # Find next target point in reference path
                current_pos = env.position
                while (current_path_idx < len(reference_path) - 1 and 
                       np.linalg.norm(np.array(current_pos) - np.array(reference_path[current_path_idx])) < 1.0):
                    current_path_idx += 1
                
                target_pos = reference_path[current_path_idx]
                direction = np.array(target_pos) - current_pos
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                
                # Convert direction to tensor before blending
                guided_action = torch.FloatTensor(np.pad(direction, (0, 1), 'constant'))
                policy_action = agent.select_action(state_tensor, eval_mode=False)
                guidance_weight = max(0, 1 - episode / (num_episodes * 0.3))
                # Ensure the blended action remains a PyTorch tensor
                action = torch.FloatTensor(guidance_weight * guided_action.numpy() + (1 - guidance_weight) * policy_action.numpy())
            else:
                action = agent.select_action(state_tensor, eval_mode=False)
            
            # Convert action to numpy before passing to env.step()
            action_np = action.detach().numpy()
            next_state, reward, done, _ = env.step(action_np)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            positions.append(env.position.copy())
            
            # Update state and counters
            state = next_state
            episode_reward += reward
            steps += 1
            steps_since_update += 1
            
            # Update policy if enough steps collected
            if steps_since_update >= num_steps_per_update:
                states, actions, old_log_probs, returns, advantages = agent.process_transitions()
                if states is not None and len(advantages) > 0:
                    loss_stats = agent.update(states, actions, old_log_probs, returns, advantages)
                    steps_since_update = 0
                    
                    if loss_stats:
                        wandb.log(loss_stats)
        
        # Visualization code (unchanged)
        if episode % 10 == 0:
            ax1.clear()
            ax2.clear()
            
            positions = np.array(positions)
            ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', label='Path')
            ax1.scatter(*env.goal, color='red', s=100, label='Goal')
            
            for obs in env.obstacles:
                x, y, z, r = obs
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                x_surf = r * np.outer(np.cos(u), np.sin(v)) + x
                y_surf = r * np.outer(np.sin(u), np.sin(v)) + y
                z_surf = r * np.outer(np.ones(np.size(u)), np.cos(v)) + z
                ax1.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.3)
            
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            ax1.set_title(f'Episode {episode} Path')
            
            episode_rewards.append(episode_reward)
            ax2.plot(episode_rewards)
            ax2.set_title('Training Rewards')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Reward')
            
            plt.draw()
            plt.pause(0.01)
        
        # Logging
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        
        wandb.log({
            "episode": episode,
            "episode_reward": episode_reward,
            "average_reward": avg_reward,
            "episode_length": steps,
            "total_steps": episode * max_steps_per_episode + steps
        })
        
        print(f"Episode {episode} | Reward: {episode_reward:.2f} | Average: {avg_reward:.2f} | Steps: {steps}")
        
        # Save best model
        if avg_reward > best_reward:
            best_reward = avg_reward
            agent.save(save_dir / "best_model.pth")
            wandb.log({"best_reward": best_reward})
    
    plt.ioff()
    wandb.finish()
    return agent, env

if __name__ == "__main__":
    train()