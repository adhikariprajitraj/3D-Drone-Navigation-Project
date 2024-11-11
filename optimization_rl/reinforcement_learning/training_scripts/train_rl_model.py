import os
import numpy as np
from datetime import datetime
from pathlib import Path
import torch
import wandb
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(project_root)
from optimization_rl.reinforcement_learning.ppo_agent import PPO, DroneEnv
from utils.logging_utils import Logger

def train():
    # Initialize logger
    logger = Logger(name="PPOTraining", log_dir="logs")
    logger.info("Starting PPO training")

    # Training parameters
    num_episodes = 1000
    max_steps_per_episode = 500
    eval_frequency = 50
    save_frequency = 100
    update_frequency = 20
    
    # Initialize environment with sample parameters
    env = DroneEnv(
        obstacles=[(5, 5, 5, 1)],  # Single obstacle for testing
        bounds=(10, 10, 10),
        goal=(8, 8, 8)
    )
    
    # Initialize agent
    state_dim = env.observation_space.shape[0]  # Should be 37 based on DroneEnv
    action_dim = env.action_space.shape[0]      # Should be 4 based on DroneEnv
    
    agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        clip_range=0.2,
        value_coef=1.0,
        entropy_coef=0.01,
        log_dir="logs"
    )
    
    # Initialize wandb
    wandb.init(
        project="drone-navigation",
        config={
            "algorithm": "PPO",
            "num_episodes": num_episodes,
            "max_steps": max_steps_per_episode,
            "state_dim": state_dim,
            "action_dim": action_dim
        }
    )
    
    # Create directory for saving models
    save_dir = Path("models") / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_reward = float('-inf')
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()  # This will now return the correct state format
        episode_reward = 0
        episode_steps = 0
        done = False
        
        # Add initial state logging
        logger.info(f"Episode {episode} starting position: {state[:3]}")  # Assuming first 3 values are position
        
        while not done and episode_steps < max_steps_per_episode:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get action from agent
            action = agent.select_action(state_tensor)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Add detailed logging
            if done:
                logger.info(f"Episode ended because: {info.get('terminal_reason', 'unknown')}")
                logger.info(f"Final position: {info.get('position', 'unknown')}")
                logger.info(f"Distance to goal: {info.get('distance_to_goal', 'unknown')}")
            
            # Add reward component logging (if available in info)
            if episode_steps % 100 == 0:  # Log every 100 steps
                logger.info(f"Step {episode_steps}: Position: {next_state[:3]}, Reward: {reward}")
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            # Update agent if enough transitions are collected
            if episode_steps % update_frequency == 0:
                # Get transitions from memory
                states, actions, rewards, next_states, dones, old_log_probs, _ = agent.memory.get_transitions()
                
                # Only update if we have enough transitions
                if len(states) > 0:  # Changed condition to ensure we have data
                    # Calculate returns and advantages
                    returns = agent.compute_returns(rewards, dones)
                    if len(returns) > 0:  # Add this check
                        advantages = agent.compute_advantages(returns, states)
                        if advantages is not None and advantages.numel() > 0:
                            # Normalize advantages
                            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                            
                            # Update policy
                            loss_stats = agent.update(states, actions, old_log_probs, returns, advantages)
                            
                            if loss_stats:
                                wandb.log({
                                    "actor_loss": loss_stats["policy_loss"],
                                    "critic_loss": loss_stats["value_loss"],
                                    "entropy": loss_stats["entropy"]
                                })
                    
                    # Clear memory after update attempt
                    agent.memory.clear()
        
        # End of episode logging
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-100:])
        
        wandb.log({
            "episode": episode,
            "episode_reward": episode_reward,
            "average_reward": avg_reward,
            "episode_steps": episode_steps
        })
        
        logger.info(f"Episode {episode}: Reward = {episode_reward:.2f}, Steps = {episode_steps}")
        
        # Save best model
        if avg_reward > best_reward:
            best_reward = avg_reward
            agent.save(save_dir / "best_model.pth")
            logger.info(f"Saved new best model with average reward {best_reward:.2f}")
        
        # Regular checkpoint saving
        if episode % save_frequency == 0:
            agent.save(save_dir / f"checkpoint_{episode}.pth")
            
        # Evaluation
        if episode % eval_frequency == 0:
            eval_rewards = evaluate(env, agent, num_eval_episodes=5)
            mean_eval_reward = np.mean(eval_rewards)
            wandb.log({
                "eval_episode": episode,
                "eval_reward": mean_eval_reward
            })
            logger.info(f"Evaluation at episode {episode}: Mean reward = {mean_eval_reward:.2f}")
    
    # Save final model
    agent.save(save_dir / "final_model.pth")
    logger.info("Training completed. Final model saved.")
    wandb.finish()
    logger.close()

def evaluate(env, agent, num_eval_episodes=5):
    """Evaluate the agent's performance."""
    eval_rewards = []
    
    for episode in range(num_eval_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Convert state to tensor and add batch dimension
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Check for any NaN or infinite values in state
            if torch.isnan(state_tensor).any() or torch.isinf(state_tensor).any():
                print(f"\nWarning: State contains NaN or Inf values:")
                print(f"State shape: {state_tensor.shape}")
                print(f"State values: {state_tensor}")
                state_tensor = torch.nan_to_num(state_tensor, nan=0.0, posinf=10.0, neginf=-10.0)
            
            try:
                action = agent.select_action(state_tensor, eval_mode=True)
            except ValueError as e:
                print(f"\nError in episode {episode}:")
                print(f"State tensor shape: {state_tensor.shape}")
                print(f"State min/max: {state_tensor.min():.3f}/{state_tensor.max():.3f}")
                print(f"Error message: {str(e)}")
                raise
                
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            
        eval_rewards.append(episode_reward)
    
    return eval_rewards

if __name__ == "__main__":
    train()