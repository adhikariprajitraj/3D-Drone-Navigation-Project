# 3D Drone Navigation Project

## Overview
The 3D Drone Navigation project aims to design a sophisticated simulation where a drone navigates through a 3D cubic space filled with static obstacles. The primary goal is to develop a reinforcement learning (RL) model and path optimization algorithms to guide the drone in finding the shortest path from a starting point to an endpoint within this environment, while avoiding collisions.

## Project Objectives
- **Simulate a 3D environment** with static obstacles to test drone navigation.
- **Develop a virtual drone model** equipped with necessary movement capabilities.
- **Implement pathfinding algorithms** such as A* and RRT to generate initial path estimates.
- **Integrate RL algorithms** (e.g., Deep Q-Networks and Actor-Critic methods) for dynamic path optimization.
- **Evaluate the performance** of the drone based on path length, time taken, and obstacle avoidance.

## Features
- **3D Cubic Environment**: A simulated 3D space containing various static obstacles.
- **Path Planning**: Initial paths are generated using traditional algorithms.
- **Reinforcement Learning**: The drone refines its path through RL training.
- **Evaluation Metrics**: The drone’s performance is measured by path length, time to destination, and efficiency.

## Project Structure
```
3D-Drone-Navigation-Project/
|-- README.md
|-- LICENSE
|-- .gitignore
|-- docs/
|   |-- project_overview.md
|   |-- references/
|-- environment/
|   |-- static_objects_data.csv
|   |-- environment_setup.py
|-- drone_model/
|   |-- drone_simulation.py
|   |-- controllers/
|   |   |-- drone_controller.py
|-- optimization_rl/
|   |-- path_planning/
|   |   |-- a_star.py
|   |   |-- rrt.py
|   |-- reinforcement_learning/
|   |   |-- dqn_agent.py
|   |   |-- actor_critic_agent.py
|   |   |-- training_scripts/
|   |       |-- train_rl_model.py
|-- tests/
|   |-- test_environment.py
|   |-- test_rl_agent.py
|-- utils/
|   |-- data_loader.py
|   |-- metrics.py
|-- requirements.txt
|-- scripts/
|   |-- run_simulation.py
|-- results/
|   |-- performance_logs/
|   |-- graphs/
```

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

4. **Run initial tests**:
   Ensure that the setup works by running tests.
   ```bash
   python -m unittest discover tests
   ```

## Usage
- **Train the RL model**: Navigate to the `optimization_rl/reinforcement_learning/training_scripts/` directory and run the training script.
   ```bash
   python train_rl_model.py
   ```
- **Run the simulation**: Use the `scripts/run_simulation.py` to start a drone simulation.
   ```bash
   python scripts/run_simulation.py
   ```

## Contributing
Contributions are welcome. Please fork the repository and submit a pull request with clear documentation.

## License
This project is licensed under the MIT License.

## Contact
For any questions or further information, please contact [prajit.076bie029@tcioe.edu.np] or [bishal.076bie011@tcioe.edu.np].

