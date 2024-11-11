from PIL import Image
import glob
from pathlib import Path

def create_gif(image_dir, output_filename='drone_trajectory.gif'):
    # Get all PNG files in the directory
    image_files = sorted(glob.glob(str(Path(image_dir) / 'trajectory_step_*.png')))
    
    if not image_files:
        print("No trajectory images found!")
        return
        
    # Create GIF
    images = [Image.open(file) for file in image_files]
    images[0].save(
        Path(image_dir) / output_filename,
        save_all=True,
        append_images=images[1:],
        duration=200,  # milliseconds per frame
        loop=0
    )
    print(f"Created animation: {output_filename}")

if __name__ == "__main__":
    create_gif("drone_visualization") 