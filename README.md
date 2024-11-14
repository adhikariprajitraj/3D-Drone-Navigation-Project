# 3D Drone Navigation Project

## Overview
The 3D Drone Navigation project aims to design a sophisticated simulation where a drone navigates through a 3D cubic space filled with static obstacles. The primary goal is to develop a reinforcement learning (RL) model and path optimization algorithms to guide the drone in finding the shortest path from a starting point to an endpoint within this environment, while avoiding collisions.

## Project Objectives
- **Simulate a 3D environment** with static obstacles to test drone navigation.
- **Develop a virtual drone model** equipped with necessary movement capabilities.
- **Implement pathfinding algorithms** such as A* and RRT to generate initial path estimates.
- **Integrate RL algorithms** (e.g., Proximal Policy Optimization) for dynamic path optimization.
- **Evaluate the performance** of the drone based on path length, time taken, and obstacle avoidance.

## Features
- **3D Cubic Environment**: A simulated 3D space containing various static obstacles.
- **Path Planning**: Initial paths are generated using traditional algorithms.
- **Reinforcement Learning**: The drone refines its path through RL training.
- **Evaluation Metrics**: The drone’s performance is measured by path length, time to destination, and efficiency.

---

## Dynamics of Quadrotor

The quadrotor dynamics are modeled using standard translational and rotational equations of motion, accounting for forces and torques applied by the four rotors. Here are the key equations:

### Symbols

| Symbol         | Description                            |
|----------------|----------------------------------------|
| \( x, y, z \)           | Position coordinates in 3D space    |
| \( u, v, w \)           | Velocity components along \(x, y, z\)|
| \( \phi, \theta, \psi \) | Roll, pitch, and yaw angles        |
| \( \omega_1, \omega_2, \omega_3, \omega_4 \) | Rotor speeds |
| \( l \)        | Distance from the center to each rotor |
| \( b \)        | Torque constant                        |
| \( k \)        | Aerodynamic constant                   |
| \( I_x, I_y, I_z \)     | Moments of inertia along each axis |
| \( m \)        | Mass of the quadrotor                  |
| \( g \)        | Gravitational acceleration             |

### Translational Dynamics

The translational dynamics, describing the drone's movement in space, are given by Newton's second law:

\[
m \ddot{x} = -\sin \theta \cdot F
\]
\[
m \ddot{y} = \sin \phi \cos \theta \cdot F
\]
\[
m \ddot{z} = g - \cos \phi \cos \theta \cdot F
\]

where:
- \( F \) is the total thrust produced by the rotors: \( F = k(\omega_1^2 + \omega_2^2 + \omega_3^2 + \omega_4^2) \).

### Rotational Dynamics

The rotational dynamics govern the drone's orientation and are given by:

\[
I_x \ddot{\phi} = l \cdot k (\omega_2^2 - \omega_4^2)
\]
\[
I_y \ddot{\theta} = l \cdot k (\omega_3^2 - \omega_1^2)
\]
\[
I_z \ddot{\psi} = b (\omega_1^2 - \omega_2^2 + \omega_3^2 - \omega_4^2)
\]

These equations determine the roll, pitch, and yaw movements of the quadrotor in response to rotor speed adjustments.

---

## Implementation Details of the Algorithm

### Environment Setup

The environment is designed as a 3D space with customizable boundaries and static obstacles. Static obstacles are represented as spherical objects, and each has a defined position and radius within the environment.

### Pathfinding

Pathfinding uses A* or RRT (Rapidly-Exploring Random Trees) to find a preliminary path from the start point to the endpoint, avoiding static obstacles. This path serves as a guide for the RL agent.

### Reinforcement Learning (RL)

The project uses Proximal Policy Optimization (PPO) as the primary RL algorithm. The RL model refines the drone’s path by interacting with the environment, learning to navigate while avoiding obstacles and optimizing for the shortest path.

### Action and State Representation

- **Actions**: The agent’s actions correspond to the four rotor speeds, \(\omega_1\), \(\omega_2\), \(\omega_3\), and \(\omega_4\), controlling thrust and rotational dynamics.
- **State**: The state vector includes position, orientation, velocity, goal distance, and obstacle information.

### Reward Function

The reward function incentivizes the agent to reach the goal efficiently:
- **Distance Reward**: Encourages reduction in distance to the goal.
- **Goal Reward**: Provides a high reward for reaching the endpoint.
- **Collision Penalty**: Penalizes the agent for colliding with obstacles.
- **Efficiency Penalty**: Penalizes unnecessary actions or deviations from the path.

---

## Installation and Setup

### Prerequisites
Ensure you have Python 3.7+ installed and a virtual environment set up.

### Setup Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/username/3D-Drone-Navigation-Project.git
   cd 3D-Drone-Navigation-Project



## Installation and Setup

### Prerequisites
Ensure you have Python 3.7+ installed and a virtual environment set up.

### Setup Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/username/3D-Drone-Navigation-Project.git
   cd 3D-Drone-Navigation-Project
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```


## Usage
- **Train the RL model**: Navigate to the `optimization_rl/reinforcement_learning/training_scripts/` directory and run the training script.
   ```bash
   python train_rl_model.py
   ```
- **Run the model**:
   ```bash
   python main.py
   ```

## Contributing
Contributions are welcome. Please fork the repository and submit a pull request with clear documentation.

## License
This project is licensed under the MIT License.

## Contact
For any questions or further information, please contact [prajit.076bie029@tcioe.edu.np] or [bishal.076bie011@tcioe.edu.np].

