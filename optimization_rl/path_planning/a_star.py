"""
A* Path Planning Algorithm for 3D Drone Navigation
"""

import numpy as np
from heapq import heappush, heappop
from typing import List, Tuple, Set, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from scipy.interpolate import make_interp_spline
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.logging_utils import Logger
import time
from typing import Optional

class Node:
    def __init__(self, position: Tuple[float, float, float], g_cost: float = float('inf'), 
                 h_cost: float = 0):
        self.position = position
        self.g_cost = g_cost  # Cost from start to current node
        self.h_cost = h_cost  # Heuristic cost (estimated) to goal
        self.f_cost = g_cost + h_cost  # Total cost
        self.parent = None

    def __lt__(self, other):
        return self.f_cost < other.f_cost

def euclidean_distance(point1: Tuple[float, float, float], 
                      point2: Tuple[float, float, float]) -> float:
    """Calculate 3D Euclidean distance between two points."""
    return np.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

def get_neighbors(current_pos: Tuple[float, float, float], 
                 step_size: float) -> List[Tuple[float, float, float]]:
    """Generate neighboring positions in 3D space."""
    neighbors = []
    # Generate 26 neighboring positions (3D grid)
    for dx in [-step_size, 0, step_size]:
        for dy in [-step_size, 0, step_size]:
            for dz in [-step_size, 0, step_size]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                neighbor = (
                    current_pos[0] + dx,
                    current_pos[1] + dy,
                    current_pos[2] + dz
                )
                neighbors.append(neighbor)
    return neighbors

def is_valid_position(position: Tuple[float, float, float], 
                     obstacles: List[Tuple[float, float, float, float]], 
                     bounds: Tuple[float, float, float]) -> bool:
    """
    Check if position is valid (within bounds and not colliding with obstacles).
    obstacles: List of (x, y, z, radius) tuples representing spherical obstacles
    bounds: (max_x, max_y, max_z) tuple representing environment boundaries
    """
    x, y, z = position
    
    # Check boundaries
    if not (0 <= x <= bounds[0] and 0 <= y <= bounds[1] and 0 <= z <= bounds[2]):
        return False
    
    # Check obstacle collisions
    for obs_x, obs_y, obs_z, radius in obstacles:
        if euclidean_distance(position, (obs_x, obs_y, obs_z)) <= radius:
            return False
    
    return True

def a_star(start: Tuple[float, float, float],
           goal: Tuple[float, float, float],
           obstacles: List[Tuple[float, float, float, float]],
           bounds: Tuple[float, float, float],
           step_size: float = 1.0,
           max_iterations: int = 10000,
           log_dir: str = "logs") -> List[Tuple[float, float, float]]:
    """
    A* pathfinding algorithm for 3D drone navigation with logging.
    """
    # Initialize logger
    logger = Logger(name="AStar", log_dir=log_dir)
    logger.info("Initializing A* path planning")
    logger.info(f"Start position: {start}")
    logger.info(f"Goal position: {goal}")
    logger.info(f"Number of obstacles: {len(obstacles)}")
    logger.info(f"Step size: {step_size}")
    
    start_time = time.time()
    
    # Initialize start node
    start_node = Node(start, g_cost=0, h_cost=euclidean_distance(start, goal))
    
    # Initialize open and closed sets
    open_set = []
    heappush(open_set, start_node)
    closed_set: Set[Tuple[float, float, float]] = set()
    
    # Keep track of node references
    node_dict: Dict[Tuple[float, float, float], Node] = {start: start_node}
    
    iterations = 0
    nodes_explored = 0
    
    while open_set and iterations < max_iterations:
        current_node = heappop(open_set)
        current_pos = current_node.position
        
        # Log progress periodically
        if iterations % 100 == 0:
            logger.info(f"Iteration {iterations}: Exploring position {current_pos}")
            logger.info(f"Current f_cost: {current_node.f_cost:.2f}")
            logger.log_scalar("search/nodes_explored", nodes_explored, iterations)
            logger.log_scalar("search/open_set_size", len(open_set), iterations)
            logger.log_scalar("search/closed_set_size", len(closed_set), iterations)
        
        if euclidean_distance(current_pos, goal) < step_size:
            # Path found
            path = []
            current = current_node
            while current:
                path.append(current.position)
                current = current.parent
            
            path = path[::-1]
            elapsed_time = time.time() - start_time
            
            # Log success metrics
            logger.info(f"Path found in {elapsed_time:.2f} seconds")
            logger.info(f"Path length: {len(path)} nodes")
            logger.info(f"Total nodes explored: {nodes_explored}")
            logger.info(f"Final path distance: {sum(euclidean_distance(path[i], path[i+1]) for i in range(len(path)-1)):.2f}")
            
            logger.log_scalar("results/path_length", len(path), 0)
            logger.log_scalar("results/nodes_explored", nodes_explored, 0)
            logger.log_scalar("results/time_taken", elapsed_time, 0)
            
            return path
        
        closed_set.add(current_pos)
        
        # Generate and explore neighbors
        for neighbor_pos in get_neighbors(current_pos, step_size):
            if neighbor_pos in closed_set or not is_valid_position(neighbor_pos, obstacles, bounds):
                continue
            
            nodes_explored += 1
            tentative_g_cost = current_node.g_cost + euclidean_distance(current_pos, neighbor_pos)
            
            if neighbor_pos not in node_dict:
                neighbor_node = Node(
                    neighbor_pos,
                    g_cost=tentative_g_cost,
                    h_cost=euclidean_distance(neighbor_pos, goal)
                )
                node_dict[neighbor_pos] = neighbor_node
                heappush(open_set, neighbor_node)
            elif tentative_g_cost < node_dict[neighbor_pos].g_cost:
                neighbor_node = node_dict[neighbor_pos]
                neighbor_node.g_cost = tentative_g_cost
                neighbor_node.f_cost = tentative_g_cost + neighbor_node.h_cost
                
            node_dict[neighbor_pos].parent = current_node
        
        iterations += 1
    
    # Log failure
    elapsed_time = time.time() - start_time
    logger.error(f"Failed to find path after {iterations} iterations")
    logger.error(f"Time elapsed: {elapsed_time:.2f} seconds")
    logger.error(f"Nodes explored: {nodes_explored}")
    
    return []

def smooth_path(path: List[Tuple[float, float, float]], 
                obstacles: List[Tuple[float, float, float, float]],
                bounds: Tuple[float, float, float],
                interpolation_points: int = 50,
                logger: Optional[Logger] = None) -> List[Tuple[float, float, float]]:
    """Smooth the path with logging."""
    if logger is None:
        logger = Logger(name="PathSmoothing", log_dir="logs")
    
    if len(path) < 3:
        logger.warning("Path too short for smoothing")
        return path
    
    logger.info("Starting path smoothing")
    logger.info(f"Original path length: {len(path)}")
    start_time = time.time()
    
    # Convert path to numpy array for easier manipulation
    path_array = np.array(path)
    
    # Parameter for interpolation
    t = np.linspace(0, 1, len(path))
    t_smooth = np.linspace(0, 1, interpolation_points)
    
    # Create B-spline for each dimension
    splines = [
        make_interp_spline(t, path_array[:, i], k=min(3, len(path)-1)) 
        for i in range(3)
    ]
    
    # Generate smooth path points
    smooth_points = np.vstack([
        spline(t_smooth) for spline in splines
    ]).T
    
    # Validate and adjust points for obstacle avoidance
    final_path = [tuple(smooth_points[0])]  # Start with first point
    
    for i in range(1, len(smooth_points)):
        current_point = smooth_points[i]
        prev_point = np.array(final_path[-1])
        
        # Check if direct path to next point collides with obstacles
        test_points = np.linspace(prev_point, current_point, 10)
        collision = False
        
        for test_point in test_points:
            if not is_valid_position(tuple(test_point), obstacles, bounds):
                collision = True
                break
        
        if collision:
            # If collision detected, find a valid intermediate point
            direction = current_point - prev_point
            direction = direction / np.linalg.norm(direction)
            
            # Try different radii to find valid point
            for radius in np.linspace(0.5, 2.0, 10):
                perpendicular = np.array([direction[1], -direction[0], 0])
                perpendicular = perpendicular / np.linalg.norm(perpendicular)
                
                # Try points in a circle around the original path
                for angle in np.linspace(0, 2*np.pi, 12):
                    offset = radius * (np.cos(angle) * perpendicular + 
                                     np.sin(angle) * np.cross(direction, perpendicular))
                    test_point = prev_point + direction + offset
                    
                    if is_valid_position(tuple(test_point), obstacles, bounds):
                        final_path.append(tuple(test_point))
                        break
                
                if len(final_path) > len(final_path) - 1:
                    break
            
            if len(final_path) == len(final_path) - 1:
                # If no valid point found, use original path point
                final_path.append(tuple(current_point))
        else:
            final_path.append(tuple(current_point))
    
    elapsed_time = time.time() - start_time
    logger.info(f"Path smoothing completed in {elapsed_time:.2f} seconds")
    logger.info(f"Smoothed path length: {len(final_path)}")
    
    # Log path metrics
    original_distance = sum(euclidean_distance(path[i], path[i+1]) for i in range(len(path)-1))
    smoothed_distance = sum(euclidean_distance(final_path[i], final_path[i+1]) for i in range(len(final_path)-1))
    
    logger.log_scalar("smoothing/original_distance", original_distance, 0)
    logger.log_scalar("smoothing/smoothed_distance", smoothed_distance, 0)
    logger.log_scalar("smoothing/distance_reduction", original_distance - smoothed_distance, 0)
    
    return final_path

if __name__ == "__main__":
    # Initialize logger
    logger = Logger(name="AStarVisualization", log_dir="logs")
    logger.info("Starting A* path planning visualization")
    
    # Create test scenario with more space between start and goal
    start = (1, 1, 1)
    goal = (8, 8, 8)
    bounds = (10, 10, 10)
    
    # Generate fewer, more spread out obstacles
    num_obstacles = 5  # Reduced from 10
    obstacles = []
    for _ in range(num_obstacles):
        # Avoid placing obstacles too close to start or goal
        while True:
            x = random.uniform(2, bounds[0]-2)
            y = random.uniform(2, bounds[1]-2)
            z = random.uniform(2, bounds[2]-2)
            radius = random.uniform(0.5, 1.0)
            # Check if obstacle is too close to start or goal
            if (euclidean_distance((x,y,z), start) > 2 and 
                euclidean_distance((x,y,z), goal) > 2):
                obstacles.append((x, y, z, radius))
                break
    
    logger.info(f"Generated {num_obstacles} random obstacles")
    
    # Find path
    path = a_star(start, goal, obstacles, bounds)
    
    # Debug prints
    print(f"Path found: {'Yes' if path else 'No'}")
    if path:
        print(f"Path length: {len(path)}")
        print(f"First few points: {path[:3]}")
        smoothed_path = smooth_path(path, obstacles, bounds, logger=logger)
        print(f"Smoothed path length: {len(smoothed_path)}")
        logger.info("Successfully found and smoothed path")
    else:
        logger.error("Failed to find path")
        exit(1)

    # Create visualization
    fig = plt.figure(figsize=(12, 12))
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

    # Plot original path
    if path:
        path_array = np.array(path)
        ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2], 
                'b--', label='Original Path', linewidth=2)

    # Plot smoothed path
    if smoothed_path:
        smoothed_array = np.array(smoothed_path)
        ax.plot(smoothed_array[:, 0], smoothed_array[:, 1], smoothed_array[:, 2], 
                'g-', linewidth=3, label='Smoothed Path')
        
        # Add path points
        ax.scatter(smoothed_array[:, 0], smoothed_array[:, 1], smoothed_array[:, 2],
                  color='blue', s=30, alpha=0.5)

    # Plot start and goal points with larger markers
    ax.scatter(*start, color='green', s=200, label='Start')
    ax.scatter(*goal, color='red', s=200, label='Goal')

    # Improve the view
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D A* Path Planning with Obstacle Avoidance')
    
    # Set consistent axis limits
    ax.set_xlim(0, bounds[0])
    ax.set_ylim(0, bounds[1])
    ax.set_zlim(0, bounds[2])
    
    # Add grid
    ax.grid(True)
    
    # Add legend
    ax.legend()

    # Make sure the plot is displayed
    plt.tight_layout()
    
    # Save the figure before showing
    plot_path = os.path.join(logger.log_dir, "path_visualization.png")
    plt.savefig(plot_path)
    logger.info(f"Saved visualization to {plot_path}")
    
    # Show the plot and block until window is closed
    plt.show(block=True)
    
    logger.info("Visualization complete")
    logger.close()
