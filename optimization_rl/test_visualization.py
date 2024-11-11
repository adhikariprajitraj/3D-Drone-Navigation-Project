import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

def simple_visualization():
    # Create a basic 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create some sample data
    t = np.linspace(0, 10, 100)
    x = t * np.cos(t)
    y = t * np.sin(t)
    z = t
    
    # Plot the data
    ax.plot(x, y, z, label='Drone Path')
    
    # Add labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Test Visualization')
    ax.legend()
    
    # Save the plot
    save_dir = Path("test_visualization")
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / "test_plot.png")
    print(f"Saved plot to: {save_dir.absolute() / 'test_plot.png'}")
    
    # Try to display the plot
    plt.show()

if __name__ == "__main__":
    print("Starting visualization test...")
    simple_visualization()
    print("Visualization test complete!") 