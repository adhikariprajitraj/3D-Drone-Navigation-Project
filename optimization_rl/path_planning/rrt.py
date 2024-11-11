"""
RRT (Rapidly-exploring Random Tree) Path Planning Algorithm for 3D Drone Navigation
"""

import numpy as np
from typing import List, Tuple, Optional
import random
import time
from utils.logging_utils import Logger

class Node:
    def __init__(self, position: Tuple[float, float, float]):
        self.position = position
        self.parent = None
        self.children = []

class RRT:
    def __init__(self, 
                 start: Tuple[float, float, float],
                 goal: Tuple[float, float, float],
                 obstacles: List[Tuple[float, float, float, float]],
                 bounds: Tuple[float, float, float],
                 step_size: float = 0.5,
                 max_iterations: int = 5000,
                 goal_sample_rate: float = 0.2,
                 log_dir: str = "logs"):
        """
        Initialize RRT path planner.
        
        Args:
            start: Starting position (x, y, z)
            goal: Goal position (x, y, z)
            obstacles: List of obstacle positions and sizes [(x, y, z, radius), ...]
            bounds: Environment boundaries (max_x, max_y, max_z)
            step_size: Distance between nodes
            max_iterations: Maximum number of iterations
            goal_sample_rate: Probability of sampling the goal position
            log_dir: Directory for logging
        """
        self.start = Node(start)
        self.goal = Node(goal)
        self.obstacles = obstacles
        self.bounds = bounds
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_sample_rate = goal_sample_rate
        self.nodes = [self.start]
        self.goal_reached = False

        # Initialize logger
        self.logger = Logger(name="RRT", log_dir=log_dir)
        self.logger.info(f"Initializing RRT with parameters:")
        self.logger.info(f"Start: {start}, Goal: {goal}")
        self.logger.info(f"Step size: {step_size}, Max iterations: {max_iterations}")
        self.logger.info(f"Goal sample rate: {goal_sample_rate}")

    def euclidean_distance(self, p1: Tuple[float, float, float], 
                          p2: Tuple[float, float, float]) -> float:
        """Calculate 3D Euclidean distance between two points."""
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    def is_collision_free(self, p1: Tuple[float, float, float], 
                         p2: Tuple[float, float, float]) -> bool:
        """Check if path between two points is collision-free."""
        # Check multiple points along the line
        vec = np.array(p2) - np.array(p1)
        dist = np.linalg.norm(vec)
        vec_normalized = vec / dist
        
        # Check points along the path
        num_points = max(int(dist / (self.step_size * 0.5)), 5)
        for i in range(num_points):
            point = np.array(p1) + vec_normalized * (i * dist / num_points)
            if not self.is_valid_position(tuple(point)):
                return False
        return True

    def is_valid_position(self, position: Tuple[float, float, float]) -> bool:
        """Check if position is valid (within bounds and not colliding with obstacles)."""
        x, y, z = position
        
        # Check boundaries
        if not (0 <= x <= self.bounds[0] and 
                0 <= y <= self.bounds[1] and 
                0 <= z <= self.bounds[2]):
            return False
        
        # Check obstacle collisions
        for obs_x, obs_y, obs_z, radius in self.obstacles:
            if self.euclidean_distance(position, (obs_x, obs_y, obs_z)) <= radius:
                return False
        
        return True

    def sample_random_position(self) -> Tuple[float, float, float]:
        """Generate a random position in the environment."""
        if random.random() < self.goal_sample_rate:
            return self.goal.position
        
        return (
            random.uniform(0, self.bounds[0]),
            random.uniform(0, self.bounds[1]),
            random.uniform(0, self.bounds[2])
        )

    def find_nearest_node(self, position: Tuple[float, float, float]) -> Node:
        """Find the nearest node in the tree to the given position."""
        distances = [self.euclidean_distance(node.position, position) 
                    for node in self.nodes]
        return self.nodes[np.argmin(distances)]

    def steer(self, from_pos: Tuple[float, float, float], 
              to_pos: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Steer from one position toward another within step_size."""
        vec = np.array(to_pos) - np.array(from_pos)
        dist = np.linalg.norm(vec)
        
        if dist <= self.step_size:
            return to_pos
        
        vec = vec / dist * self.step_size
        return tuple(np.array(from_pos) + vec)

    def plan(self) -> Optional[List[Tuple[float, float, float]]]:
        """Execute RRT path planning with logging."""
        start_time = time.time()
        self.logger.info("Starting RRT path planning")
        
        for iteration in range(self.max_iterations):
            # Sample random position
            random_pos = self.sample_random_position()
            
            # Find nearest node
            nearest_node = self.find_nearest_node(random_pos)
            
            # Steer towards random position
            new_pos = self.steer(nearest_node.position, random_pos)
            
            # Check if new position is valid and path is collision-free
            if (self.is_valid_position(new_pos) and 
                self.is_collision_free(nearest_node.position, new_pos)):
                
                # Create new node
                new_node = Node(new_pos)
                new_node.parent = nearest_node
                nearest_node.children.append(new_node)
                self.nodes.append(new_node)
                
                # Log progress
                if iteration % 100 == 0:
                    self.logger.info(f"Iteration {iteration}: Added node at {new_pos}")
                    self.logger.log_scalar("tree_size", len(self.nodes), iteration)
                
                # Check if goal is reached
                distance_to_goal = self.euclidean_distance(new_pos, self.goal.position)
                self.logger.log_scalar("distance_to_goal", distance_to_goal, iteration)
                
                if distance_to_goal <= self.step_size:
                    self.goal.parent = new_node
                    self.goal_reached = True
                    path = self.get_path()
                    
                    elapsed_time = time.time() - start_time
                    self.logger.info(f"Path found in {elapsed_time:.2f} seconds")
                    self.logger.info(f"Path length: {len(path)} nodes")
                    self.logger.info(f"Total tree size: {len(self.nodes)} nodes")
                    
                    return path
        
        self.logger.warning("Failed to find path within maximum iterations")
        return None

    def get_path(self) -> List[Tuple[float, float, float]]:
        """Reconstruct path from goal to start."""
        if not self.goal_reached:
            return []
        
        path = [self.goal.position]
        current_node = self.goal.parent
        
        while current_node is not None:
            path.append(current_node.position)
            current_node = current_node.parent
        
        return list(reversed(path))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Create test scenario
    start = (1, 1, 1)
    goal = (8, 8, 8)
    bounds = (10, 10, 10)
    
    # Generate random obstacles
    num_obstacles = 10
    obstacles = []
    for _ in range(num_obstacles):
        x = random.uniform(0, bounds[0])
        y = random.uniform(0, bounds[1])
        z = random.uniform(0, bounds[2])
        radius = random.uniform(0.5, 1.0)
        obstacles.append((x, y, z, radius))

    # Create and execute RRT planner
    rrt = RRT(start, goal, obstacles, bounds)
    path = rrt.plan()

    # Visualization
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot obstacles
    for obs in obstacles:
        x, y, z, r = obs
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x_surf = r * np.outer(np.cos(u), np.sin(v)) + x
        y_surf = r * np.outer(np.sin(u), np.sin(v)) + y
        z_surf = r * np.outer(np.ones(np.size(u)), np.cos(v)) + z
        ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.3)

    # Plot all nodes and connections
    for node in rrt.nodes:
        if node.parent is not None:
            ax.plot([node.position[0], node.parent.position[0]],
                   [node.position[1], node.parent.position[1]],
                   [node.position[2], node.parent.position[2]],
                   'gray', alpha=0.3)

    # Plot final path
    if path:
        path_array = np.array(path)
        ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2], 
                'g-', linewidth=2, label='Path')

    # Plot start and goal points
    ax.scatter(*start, color='green', s=100, label='Start')
    ax.scatter(*goal, color='red', s=100, label='Goal')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D RRT Path Planning')
    
    # Set axis limits
    ax.set_xlim(0, bounds[0])
    ax.set_ylim(0, bounds[1])
    ax.set_zlim(0, bounds[2])
    
    # Add legend
    ax.legend()

    plt.show()
