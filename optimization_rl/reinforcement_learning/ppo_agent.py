""" 
PPO Agent for 3D Drone Navigation
"""

import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple, List, Dict
from utils.logging_utils import Logger
from utils.visualization import visualize_trajectory
import time

class DroneEnv(gym.Env):
    """Custom Environment for 3D Drone Navigation"""
    
    def __init__(self, 
                 obstacles: List[Tuple[float, float, float, float]],
                 bounds: Tuple[float, float, float],
                 goal: Tuple[float, float, float]):
        super().__init__()
        
        # Environment parameters
        self.obstacles = obstacles
        self.bounds = bounds
        self.goal = goal
        
        # Drone parameters
        self.l = 0.5  # Arm length (example value)
        self.b = 1e-7  # Torque constant (example value)
        self.k = 1e-6  # Lift constant (example value)
        self.I_x3 = 0.01  # Inertia around x-axis
        self.I_y3 = 0.01  # Inertia around y-axis
        self.I_z3 = 0.02  # Inertia around z-axis
        self.m = 1.5  # Mass of the drone
        self.g = 9.81  # Gravity
        self.dt = 0.1  # Time step
        
        # Add angular velocity initialization
        self.p = 0.0  # Roll rate
        self.q = 0.0  # Pitch rate
        self.r = 0.0  # Yaw rate
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low = 0.0,
            high = 1.0,
            shape=(4,),  # [w1, w2, w3, w4 : rotor speeds]
            dtype=np.float32
        )
        
        # State space: [position(3), orientation(3), velocity(3), angular_velocity(3), goal_distance(1), obstacle_info(27)]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(40,),  # Updated from 37 to 40 to include p, q, r
            dtype=np.float32
        )
        
        # Initialize drone state
        self.reset()

    def reset(self):
        """Reset environment to initial state"""
        # Initialize drone position with some randomization but closer to start
        self.position = np.array([
            np.random.uniform(0.0, 2.0),
            np.random.uniform(0.0, 2.0),
            np.random.uniform(0.0, 2.0)
        ])
        self.orientation = np.zeros(3)  # Start with stable orientation
        self.velocity = np.zeros(3)     # Linear velocity
        
        # Reset angular velocities
        self.p = 0.0  # Roll rate
        self.q = 0.0  # Pitch rate
        self.r = 0.0  # Yaw rate
        
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Construct state vector with angular velocities"""
        # Calculate goal distance
        goal_distance = np.linalg.norm(self.position - np.array(self.goal))
        
        # Get obstacle information
        obstacle_info = self._get_obstacle_info()
        
        # Create angular velocity vector
        angular_velocity = np.array([self.p, self.q, self.r])
        
        # Normalize angular velocities (optional but recommended)
        max_angular_velocity = 2 * np.pi  # Maximum expected angular velocity in rad/s
        normalized_angular_velocity = np.clip(angular_velocity / max_angular_velocity, -1, 1)
        
        # Concatenate all state components
        state = np.concatenate([
            self.position,          # [0:3]   - Position (x, y, z)
            self.orientation,       # [3:6]   - Orientation (roll, pitch, yaw)
            self.velocity,          # [6:9]   - Linear velocity
            normalized_angular_velocity,  # [9:12]  - Angular velocity (p, q, r)
            [goal_distance],        # [12]    - Distance to goal
            obstacle_info          # [13:40] - Obstacle information
        ])
        
        return state

    def _get_obstacle_info(self) -> np.ndarray:
        """Get simplified obstacle information from surrounding voxels"""
        # Simple 3x3x3 grid around drone
        obstacle_info = np.zeros(27)
        current_idx = 0
        
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                for z in [-1, 0, 1]:
                    point = self.position + np.array([x, y, z])
                    obstacle_info[current_idx] = self._check_point_collision(point)
                    current_idx += 1
        
        return obstacle_info

    def _check_point_collision(self, point: np.ndarray) -> float:
        """Check if point collides with any obstacle"""
        min_distance = float('inf')
        for obs_x, obs_y, obs_z, radius in self.obstacles:
            distance = np.linalg.norm(point - np.array([obs_x, obs_y, obs_z]))
            min_distance = min(min_distance, distance - radius)
        return min_distance

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one environment step"""
        # Update drone state based on action
        self._apply_action(action)
        
        # Get new state
        new_state = self._get_state()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self._is_done()
        
        return new_state, reward, done, {}

    def _apply_action(self, action: np.ndarray):
        # Unpack rotor speeds from action
        omega1, omega2, omega3, omega4 = action

        # Calculate total thrust
        T = self.k * (omega1**2 + omega2**2 + omega3**2 + omega4**2)

        # Calculate torques
        tau_x = self.l * self.k * (omega4**2 - omega2**2)  # Roll torque
        tau_y = self.l * self.k * (omega3**2 - omega1**2)  # Pitch torque
        tau_z = self.b * (omega1**2 - omega2**2 + omega3**2 - omega4**2)  # Yaw torque

        # Translational accelerations
        a_x = T * np.sin(self.orientation[1]) / self.m
        a_y = T * np.sin(self.orientation[0]) * np.cos(self.orientation[1]) / self.m
        a_z = (T - self.m * self.g) / self.m

        # Rotational accelerations (using Newton-Euler equations)
        p_dot = (tau_x - self.q * self.r * (self.I_z3 - self.I_y3)) / self.I_x3
        q_dot = (tau_y - self.r * self.p * (self.I_x3 - self.I_z3)) / self.I_y3
        r_dot = (tau_z - self.p * self.q * (self.I_y3 - self.I_x3)) / self.I_z3

        # Update velocities
        self.velocity[0] += a_x * self.dt
        self.velocity[1] += a_y * self.dt
        self.velocity[2] += a_z * self.dt

        # Update angular velocities
        self.p += p_dot * self.dt
        self.q += q_dot * self.dt
        self.r += r_dot * self.dt

        # Update position based on new velocity
        self.position += self.velocity * self.dt

        # Update orientation based on new angular velocity
        self.orientation[0] += self.p * self.dt  # Roll
        self.orientation[1] += self.q * self.dt  # Pitch
        self.orientation[2] += self.r * self.dt  # Yaw


    def _is_valid_position(self, position: np.ndarray) -> bool:
        """Check if position is valid (within bounds and not colliding)"""
        # Check bounds
        for i, pos in enumerate(position):
            if pos < 0 or pos > self.bounds[i]:
                return False
        
        # Check obstacle collision
        if self._check_point_collision(position) <= 0:
            return False
        
        return True

    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate reward based on current state including angular velocity"""
        reward = 0.0
        
        # Distance to goal reward
        current_distance = np.linalg.norm(self.position - np.array(self.goal))
        previous_distance = np.linalg.norm(self.position - self.velocity - np.array(self.goal))
        
        # Reward for moving towards the goal
        distance_reward = (previous_distance - current_distance) * 10.0  # Increased multiplier
        reward += distance_reward
        
        # Goal reached reward (increased)
        if current_distance < 0.5:
            reward += 1000.0  # Much bigger reward for reaching goal
        
        # Collision penalty (adjusted)
        if self._check_point_collision(self.position) <= 0.2:
            reward -= 100.0  # Reduced penalty
        
        # Small step penalty to encourage efficiency
        reward -= 0.1  # Reduced step penalty
        
        # Stability reward: penalize excessive rotations
        orientation_penalty = -0.1 * np.sum(np.abs(self.orientation))
        reward += orientation_penalty
        
        # Add penalty for excessive angular velocities
        angular_velocity_magnitude = np.sqrt(self.p**2 + self.q**2 + self.r**2)
        max_desired_angular_velocity = np.pi  # Maximum desired angular velocity in rad/s
        if angular_velocity_magnitude > max_desired_angular_velocity:
            angular_velocity_penalty = -0.1 * (angular_velocity_magnitude - max_desired_angular_velocity)
            reward += angular_velocity_penalty
        
        # Rotor speed penalties based on action values (since actions are rotor speeds)
        max_speed = 1.0  # Define maximum allowed rotor speed
        for speed in action:  # Assume action contains the four rotor speeds
            if speed < 0.0:
                reward -= 50.0  # Penalty for negative speeds
            elif speed > max_speed:
                reward -= (speed - max_speed) * 10.0  # Penalty for exceeding max speed

        return reward


    def _is_done(self) -> bool:
        """Check if episode is complete"""
        # Goal reached
        if np.linalg.norm(self.position - np.array(self.goal)) < 0.5:
            return True
        
        # Collision
        if self._check_point_collision(self.position) <= 0.1:
            return True
        
        return False

    def _extract_angular_velocities(self, state: np.ndarray) -> np.ndarray:
        """Extract angular velocities from state vector"""
        # Angular velocities are at indices 9:12
        normalized_angular_velocity = state[9:12]
        
        # Denormalize if necessary
        max_angular_velocity = 2 * np.pi
        angular_velocity = normalized_angular_velocity * max_angular_velocity
        
        return angular_velocity

class ActorCritic(nn.Module):
    """Actor-Critic Network for PPO"""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        
        # Shared features extractor
        self.features = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),  # Outputs mean of action distribution
            nn.Softplus()
        )
        
        # Learnable log standard deviation
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network"""
        # Add state normalization
        state = (state - state.mean()) / (state.std() + 1e-8)
        
        features = self.features(state)
        
        # Get action mean from actor network
        action_mean = self.actor(features)
        
        # Get action std from learnable parameter
        action_std = torch.clamp(self.actor_log_std.exp(), min=1e-3, max=1.0)
        
        # Get value from critic network
        value = self.critic(features)
        
        return action_mean, action_std, value

    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from the policy, ensuring non-negative rotor speeds."""
        action_mean, action_std, _ = self(state)
        action_mean = torch.relu(action_mean)  # Apply ReLU to ensure non-negative mean
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        action = torch.clamp(action, min=0.0)  # Clamp to ensure non-negative rotor speeds
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob



    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get the value estimate for a given state"""
        features = self.features(state)
        return self.critic(features)

    def get_action_mean(self, state: torch.Tensor) -> torch.Tensor:
        """Get the mean action for a given state"""
        features = self.features(state)
        return self.actor(features)

class Memory:
    """Memory buffer for storing trajectories"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def store(self, state, action, reward, next_state, done, log_prob=None, value=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        if log_prob is not None:
            self.log_probs.append(log_prob)
        if value is not None:
            self.values.append(value)

    def get_transitions(self):
        """Convert and concatenate stored transitions into tensors"""
        # Convert lists to tensors, ensuring proper type conversion
        states = torch.cat([s if isinstance(s, torch.Tensor) else torch.FloatTensor(s) 
                          for s in self.states])
        actions = torch.cat([a if isinstance(a, torch.Tensor) else torch.FloatTensor(a) 
                           for a in self.actions])
        rewards = torch.cat([torch.FloatTensor([r]) if not isinstance(r, torch.Tensor) 
                           else r for r in self.rewards])
        next_states = torch.cat([torch.FloatTensor(s) if not isinstance(s, torch.Tensor) 
                               else s for s in self.next_states])
        dones = torch.cat([torch.FloatTensor([d]) if not isinstance(d, torch.Tensor) 
                          else d for d in self.dones])
        log_probs = torch.cat(self.log_probs)
        values = torch.cat(self.values)
        
        # Clear memory
        self.clear()
        
        return states, actions, rewards, next_states, dones, log_probs, values

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

class PPO(torch.nn.Module):
    """PPO Agent"""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 clip_range: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 log_dir: str = "logs"):
        
        super(PPO, self).__init__()
        
        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(),
            lr=lr,
            eps=1e-5  # Increased epsilon for numerical stability
        )
        
        self.gamma = gamma
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Initialize logger
        self.logger = Logger(name="PPO", log_dir=log_dir)
        self.logger.info(f"Initializing PPO agent with parameters:")
        self.logger.info(f"State dim: {state_dim}, Action dim: {action_dim}")
        self.logger.info(f"Learning rate: {lr}, Gamma: {gamma}")
        self.logger.info(f"Clip range: {clip_range}")
        self.logger.info(f"Value coefficient: {value_coef}")
        self.logger.info(f"Entropy coefficient: {entropy_coef}")

        self._update_step = 0

        # Add memory
        self.memory = Memory()

        # Add these parameters
        self.gae_lambda = 0.95  # GAE parameter

        # Make sure these lines exist in your __init__
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Add max gradient norm
        self.max_grad_norm = 0.5

    def update(self, 
               states: torch.Tensor,
               actions: torch.Tensor,
               old_log_probs: torch.Tensor,
               returns: torch.Tensor,
               advantages: torch.Tensor) -> Dict[str, float]:
        """Update policy and value function with logging"""
        
        # Add check for NaN values
        if torch.isnan(states).any():
            self.logger.warning("NaN values detected in states")
            return None
            
        # Get current policy outputs
        action_mean, action_std, values = self.actor_critic(states)
        
        # Add check for NaN values
        if torch.isnan(action_mean).any() or torch.isnan(action_std).any():
            self.logger.warning("NaN values detected in policy output")
            return None
            
        dist = Normal(action_mean, action_std)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().mean()
        
        # Policy loss with clipping
        ratio = (new_log_probs - old_log_probs).exp()
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        
        # Value loss with clipping
        value_pred = values.squeeze()
        value_loss = 0.5 * (returns - value_pred).pow(2).mean()
        
        # Total loss
        loss = (policy_loss + 
                self.value_coef * value_loss - 
                self.entropy_coef * entropy)
        
        # Update network with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Log training metrics
        metrics = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'mean_advantage': advantages.mean().item(),
            'mean_return': returns.mean().item(),
            'mean_value': values.mean().item()
        }
        
        for name, value in metrics.items():
            self.logger.log_scalar(f"training/{name}", value, self._update_step)
        
        self._update_step += 1

        # Store losses for logging
        self.policy_loss = policy_loss.item()
        self.value_loss = value_loss.item()

        return metrics

    def select_action(self, state, eval_mode=False):
        """Select action from the policy"""
        # Convert state to tensor if it's not already
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            if eval_mode:
                action_mean, _, _ = self.actor_critic(state)
                return action_mean.detach().numpy().squeeze()
            else:
                action, _ = self.actor_critic.get_action(state)
                return action.detach().numpy().squeeze()

    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in memory"""
        # Ensure state is a tensor
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        
        # Convert action to tensor if it's not already
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()
        
        # Convert next_state to tensor
        if isinstance(next_state, np.ndarray):
            next_state = torch.from_numpy(next_state).float()
        
        # Get log probability and value for the state-action pair
        with torch.no_grad():
            dist = self.get_action_distribution(state)
            log_prob = dist.log_prob(action).sum(dim=-1)
            value = self.get_value(state)
        
        # Store everything in memory
        self.memory.states.append(state)
        self.memory.actions.append(action)
        self.memory.rewards.append(torch.FloatTensor([reward]))
        self.memory.next_states.append(next_state)
        self.memory.dones.append(torch.FloatTensor([done]))
        self.memory.log_probs.append(log_prob)
        self.memory.values.append(value)

    def compute_returns(self, rewards, dones):
        """Compute returns using GAE"""
        if len(rewards) == 0:
            return torch.FloatTensor([])

        returns = []
        gae = 0
        next_value = 0  # For last step
        values = self.memory.values

        if not values:
            return torch.FloatTensor([])

        # Convert to numpy arrays for easier manipulation
        rewards = np.array(rewards)
        dones = np.array(dones)
        values = np.array(values)

        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[step + 1]
                
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])

        return torch.FloatTensor(returns)

    def compute_advantages(self, returns, states):
        """Compute advantages"""
        if len(returns) == 0:
            return torch.FloatTensor([])

        with torch.no_grad():
            _, _, values = self.actor_critic(states)
        
        advantages = returns - values.squeeze()
        return advantages

    def get_log_probs(self, states, actions):
        """Get log probabilities of actions"""
        action_mean, action_std, _ = self.actor_critic(states)
        dist = Normal(action_mean, action_std)
        return dist.log_prob(actions).sum(dim=-1)

    def save(self, path):
        """Save the model's state dictionaries to the specified path."""
        torch.save({
            'actor_state_dict': self.actor_critic.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        """Load the model's state dictionaries from the specified path."""
        checkpoint = torch.load(path)
        self.actor_critic.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def process_transitions(self):
        """Process stored transitions and prepare data for update"""
        # Get stored transitions
        states, actions, rewards, next_states, dones, old_log_probs, values = self.memory.get_transitions()
        
        if len(rewards) == 0:
            return None, None, None, None, None

        # Compute returns and advantages
        returns = self.compute_returns(rewards, dones)
        advantages = self.compute_advantages(returns, states)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return states, actions, old_log_probs, returns, advantages

    def get_action_distribution(self, state):
        """Get the action distribution for a given state"""
        # Get mean and std from actor network
        action_mean, action_std, _ = self.actor_critic(state)
        
        # Create normal distribution
        dist = torch.distributions.Normal(action_mean, action_std)
        
        return dist

    def get_value(self, state):
        """Get the value estimate for a given state"""
        return self.actor_critic.get_value(state)

if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create environment
    obstacles = [(5, 5, 5, 1)]  # Single obstacle for testing
    bounds = (10, 10, 10)
    goal = (8, 8, 8)
    
    env = DroneEnv(obstacles, bounds, goal)
    
    # Create PPO agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPO(state_dim, action_dim)
    
    # Training loop
    episodes = 1000
    max_steps = 200
    
    # Initialize logger
    logger = Logger(name="DroneTraining", log_dir="logs")
    
    # Training metrics
    best_reward = float('-inf')
    episode_rewards = []
    start_time = time.time()
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        for step in range(max_steps):
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get action from policy
            with torch.no_grad():
                action, _ = agent.get_action(state_tensor)
            action = action.numpy().squeeze()
            
            # Take step in environment
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_steps += 1
            
            if done:
                break
            
            state = next_state
        
        # Log episode metrics
        episode_rewards.append(episode_reward)
        logger.log_scalar("episode/reward", episode_reward, episode)
        logger.log_scalar("episode/steps", episode_steps, episode)
        
        # Calculate and log moving averages
        if len(episode_rewards) >= 100:
            avg_reward = np.mean(episode_rewards[-100:])
            logger.log_scalar("episode/avg_reward_100", avg_reward, episode)
            
            if avg_reward > best_reward:
                best_reward = avg_reward
                logger.info(f"New best average reward: {best_reward:.2f}")
        
        if episode % 100 == 0:
            elapsed_time = time.time() - start_time
            logger.info(f"Episode {episode}/{episodes}")
            logger.info(f"Average reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
            logger.info(f"Time elapsed: {elapsed_time:.2f}s")
            
            # Log trajectory visualization
            fig = visualize_trajectory(env, agent)
            logger.log_figure("trajectory", fig, episode)
            plt.close(fig)
    
    logger.info("Training completed!")
    logger.info(f"Best average reward: {best_reward:.2f}")
    logger.close() 