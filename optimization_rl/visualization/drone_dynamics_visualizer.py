import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import os
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

class DronePhysics:
    def __init__(self):
        # Physical parameters (matching DroneEnv)
        self.m = 2.0  # Mass (kg)
        self.l = 0.25  # Arm length (m)
        self.k = 3.13e-5  # Thrust coefficient
        self.b = 7.5e-7   # Moment coefficient
        self.g = 9.81     # Gravity (m/s^2)
        
        # Moments of inertia
        self.I_x3 = 0.00232  # kg*m^2
        self.I_y3 = 0.00232  # kg*m^2
        self.I_z3 = 0.00468  # kg*m^2
        
        # Time step
        self.dt = 0.01  # Match visualizer time step
        
        # State initialization
        self.position = np.zeros(3)    # x, y, z
        self.velocity = np.zeros(3)    # vx, vy, vz
        self.orientation = np.zeros(3)  # roll, pitch, yaw
        self.p = 0.0  # Angular velocity around x
        self.q = 0.0  # Angular velocity around y
        self.r = 0.0  # Angular velocity around z
    
    def rotation_matrix(self, angles):
        """Calculate rotation matrix from euler angles (roll, pitch, yaw)"""
        phi, theta, psi = angles
        
        # Elementary rotation matrices
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
        ])
        
        R_y = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        
        R_z = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])
        
        # Complete rotation matrix
        R = R_z @ R_y @ R_x
        return R
    
    def step(self, state, rotor_speeds):
        """Update state using drone dynamics"""
        # Unpack state
        self.position = state[0:3]
        self.velocity = state[3:6]
        self.orientation = state[6:9]
        self.p = state[9]
        self.q = state[10]
        self.r = state[11]
        
        # Unpack rotor speeds
        omega1, omega2, omega3, omega4 = rotor_speeds
        
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
        
        # Update position
        self.position += self.velocity * self.dt
        
        # Update orientation
        self.orientation[0] += self.p * self.dt  # Roll
        self.orientation[1] += self.q * self.dt  # Pitch
        self.orientation[2] += self.r * self.dt  # Yaw
        
        # Pack new state
        new_state = np.zeros(12)
        new_state[0:3] = self.position
        new_state[3:6] = self.velocity
        new_state[6:9] = self.orientation
        new_state[9] = self.p
        new_state[10] = self.q
        new_state[11] = self.r
        
        return new_state

class DroneDynamicsVisualizer:
    def __init__(self):
        # Initialize drone physics
        self.drone = DronePhysics()
        
        # Timing parameters
        self.dt = 0.01  # 100Hz simulation
        self.last_update_time = time.time()
        self.sim_time = 0
        self.time_multiplier = 1.0
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Drone Dynamics Visualizer")
        self.root.geometry("1400x800")
        
        # Create frames
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(side="left", fill="y", padx=10, pady=10)
        
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # Create figure for plotting
        self.fig = plt.figure(figsize=(12, 8))
        self.ax1 = self.fig.add_subplot(131, projection='3d')
        self.ax2 = self.fig.add_subplot(132)
        self.ax3 = self.fig.add_subplot(133)
        
        # Embed matplotlib figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Create all controls including time controls
        self.create_controls()
        self.create_time_controls()
        
        # Create telemetry display
        self.create_telemetry_display()
        
        # Animation parameters
        self.frame_count = 1000
        self.is_running = False
        self.update_interval = 20
        
        # Initialize simulation state
        self.reset_simulation()
        
        # Start update loop
        self.update_loop()
    
    def create_telemetry_display(self):
        """Create telemetry display panel"""
        telemetry_frame = ttk.LabelFrame(self.control_frame, text="Telemetry")
        telemetry_frame.pack(fill="x", padx=5, pady=5)
        
        self.position_label = ttk.Label(telemetry_frame, text="Position: [0, 0, 0]")
        self.position_label.pack()
        
        self.orientation_label = ttk.Label(telemetry_frame, text="Orientation: [0, 0, 0]")
        self.orientation_label.pack()
        
        self.velocity_label = ttk.Label(telemetry_frame, text="Velocity: [0, 0, 0]")
        self.velocity_label.pack()
        
        self.time_display_label = ttk.Label(telemetry_frame, 
            text=f"Simulation Speed: {self.time_multiplier:.1f}x\nSim Time: {self.sim_time:.2f}s")
        self.time_display_label.pack()
    
    def update_telemetry_display(self):
        """Update telemetry display"""
        self.position_label.config(
            text=f"Position: [{self.state[0]:.2f}, {self.state[1]:.2f}, {self.state[2]:.2f}]")
        self.orientation_label.config(
            text=f"Orientation: [{np.rad2deg(self.state[6]):.1f}°, {np.rad2deg(self.state[7]):.1f}°, {np.rad2deg(self.state[8]):.1f}°]")
        self.velocity_label.config(
            text=f"Velocity: [{self.state[3]:.2f}, {self.state[4]:.2f}, {self.state[5]:.2f}]")
        self.time_display_label.config(
            text=f"Simulation Speed: {self.time_multiplier:.1f}x\nSim Time: {self.sim_time:.2f}s")
    
    def update_loop(self):
        """Main update loop with improved timing"""
        if self.is_running:
            current_time = time.time()
            elapsed_time = current_time - self.last_update_time
            
            # Calculate number of physics steps needed
            steps_needed = int((elapsed_time * self.time_multiplier) / self.dt)
            
            if steps_needed > 0:
                # Get current rotor speeds
                rotor_speeds = np.array([
                    self.w1_slider.get(),
                    self.w2_slider.get(),
                    self.w3_slider.get(),
                    self.w4_slider.get()
                ])
                
                # Update physics multiple times if needed
                for _ in range(min(steps_needed, 10)):  # Limit max steps to prevent spiral of death
                    self.state = self.drone.step(self.state, rotor_speeds)
                    self.sim_time += self.dt
                
                # Update visualization
                self.update_visualization()
                self.last_update_time = current_time
        
        # Schedule next update with fixed interval
        self.root.after(10, self.update_loop)  # 100Hz target update rate
    
    def update_visualization(self):
        """Update all plots"""
        # Store complete history
        self.position_history.append(self.state[:3].copy())
        self.orientation_history.append(self.state[6:9].copy())
        
        # Get current position
        current_pos = self.state[:3]
        
        # Calculate adaptive view bounds
        recent_positions = np.array(self.position_history[-300:])  # Last 3s at 100Hz
        margin = 3.0
        x_min = min(recent_positions[:, 0].min() - margin, 0)
        x_max = max(recent_positions[:, 0].max() + margin, 0)
        y_min = min(recent_positions[:, 1].min() - margin, 0)
        y_max = max(recent_positions[:, 1].max() + margin, 0)
        z_min = 0
        z_max = max(recent_positions[:, 2].max() + margin, 5)
        
        # Clear and update 3D plot
        self.ax1.clear()
        
        # Plot trajectory
        if len(self.position_history) > 1:
            pos_history = np.array(self.position_history)
            self.ax1.plot3D(pos_history[:, 0], 
                          pos_history[:, 1], 
                          pos_history[:, 2], 
                          'b-', alpha=0.5, linewidth=1)
            
            # Plot current position
            self.ax1.scatter(*current_pos, color='red', s=100)
        
        # Draw drone
        self.draw_drone(self.ax1, current_pos, self.state[6:9])
        
        # Draw ground plane grid
        x_grid = np.linspace(x_min, x_max, 10)
        y_grid = np.linspace(y_min, y_max, 10)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.zeros_like(X)
        self.ax1.plot_surface(X, Y, Z, alpha=0.1, color='gray')
        
        # Update plot settings
        self.ax1.set_xlim([x_min, x_max])
        self.ax1.set_ylim([y_min, y_max])
        self.ax1.set_zlim([z_min, z_max])
        self.ax1.set_box_aspect([1,1,1])
        
        # Labels
        self.ax1.set_xlabel('X (m)')
        self.ax1.set_ylabel('Y (m)')
        self.ax1.set_zlabel('Z (m)')
        self.ax1.set_title('Drone Position')
        
        # Update telemetry plots
        self.update_telemetry_plots()
        
        # Refresh canvas
        self.fig.canvas.draw_idle()
        
        # Update telemetry display
        self.update_telemetry_display()
    
    def update_telemetry_plots(self):
        """Update position and orientation plots"""
        # Clear plots
        self.ax2.clear()
        self.ax3.clear()
        
        if len(self.position_history) > 1:
            pos_history = np.array(self.position_history)
            time_points = np.linspace(0, self.sim_time, len(pos_history))
            
            # Plot complete position history
            self.ax2.plot(time_points, pos_history[:, 0], 'r-', label='X', alpha=0.8)
            self.ax2.plot(time_points, pos_history[:, 1], 'g-', label='Y', alpha=0.8)
            self.ax2.plot(time_points, pos_history[:, 2], 'b-', label='Z', alpha=0.8)
            
            # Show only last 10 seconds in the view
            if self.sim_time > 10:
                self.ax2.set_xlim([self.sim_time - 10, self.sim_time])
            
            self.ax2.legend()
            self.ax2.set_xlabel('Time (s)')
            self.ax2.set_ylabel('Position (m)')
            self.ax2.grid(True)
            
            # Plot complete orientation history
            orient_history = np.array(self.orientation_history)
            self.ax3.plot(time_points, np.rad2deg(orient_history[:, 0]), 'r-', label='Roll', alpha=0.8)
            self.ax3.plot(time_points, np.rad2deg(orient_history[:, 1]), 'g-', label='Pitch', alpha=0.8)
            self.ax3.plot(time_points, np.rad2deg(orient_history[:, 2]), 'b-', label='Yaw', alpha=0.8)
            
            # Show only last 10 seconds in the view
            if self.sim_time > 10:
                self.ax3.set_xlim([self.sim_time - 10, self.sim_time])
            
            self.ax3.legend()
            self.ax3.set_xlabel('Time (s)')
            self.ax3.set_ylabel('Angle (deg)')
            self.ax3.grid(True)
    
    def reset_simulation(self):
        """Reset simulation state"""
        # Reset state
        self.state = np.zeros(12)
        self.state[2] = 1.0  # Start at z=1
        
        # Reset history
        self.position_history = []
        self.orientation_history = []
        
        # Clear plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        # Initial visualization
        self.update_visualization()
        
        # Reset sliders to hover position
        hover_speed = np.sqrt(self.drone.m * self.drone.g / (4 * self.drone.k))
        for slider in [self.w1_slider, self.w2_slider, self.w3_slider, self.w4_slider]:
            slider.set(hover_speed)
        
        # Update display
        self.update_telemetry_display()
        
        # Reset timing
        self.sim_time = 0
        self.last_update_time = time.time()
    
    def create_controls(self):
        """Create all control elements"""
        # Title for controls
        ttk.Label(self.control_frame, text="Rotor Controls", 
                 style='Title.TLabel').pack(pady=20)
        
        # Create sliders with range from 0 to 2000
        self.w1_slider = self.create_slider("Rotor 1 (Front Right)", 0, 2000, width=400)
        self.w2_slider = self.create_slider("Rotor 2 (Front Left)", 0, 2000, width=400)
        self.w3_slider = self.create_slider("Rotor 3 (Back Left)", 0, 2000, width=400)
        self.w4_slider = self.create_slider("Rotor 4 (Back Right)", 0, 2000, width=400)
        
        # Add hover and zero buttons
        button_frame = ttk.Frame(self.control_frame)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Set Hover Speed", 
                  style='Large.TButton', width=20,
                  command=self.set_hover_speed).pack(side="left", padx=10)
        
        ttk.Button(button_frame, text="Zero Motors", 
                  style='Large.TButton', width=20,
                  command=self.set_zero_speed).pack(side="left", padx=10)
        
        # Add emergency stop button
        ttk.Button(self.control_frame, text="EMERGENCY STOP", 
                  style='Large.TButton', width=20,
                  command=self.emergency_stop).pack(pady=10)
        
        # Add separator
        ttk.Separator(self.control_frame, orient='horizontal').pack(fill='x', pady=20)
        
        # Simulation controls
        ttk.Label(self.control_frame, text="Simulation Controls", 
                 style='Title.TLabel').pack(pady=20)
        
        # Control buttons
        button_frame = ttk.Frame(self.control_frame)
        button_frame.pack(pady=20)
        
        for text, command in [("Start", self.start_simulation), 
                            ("Pause", self.pause_simulation),
                            ("Reset", self.reset_simulation)]:
            ttk.Button(button_frame, text=text, command=command,
                      style='Large.TButton', width=15).pack(side="left", padx=10)
        
        # Add real-time values display
        self.values_frame = ttk.LabelFrame(self.control_frame, text="Real-time Values",
                                         style='Large.TLabel')
        self.values_frame.pack(fill="x", pady=20, padx=10)
        
        # Create labels with larger font
        self.position_label = ttk.Label(self.values_frame, 
                                      text="Position: [0, 0, 0]",
                                      style='Large.TLabel')
        self.position_label.pack(pady=5)
        
        self.orientation_label = ttk.Label(self.values_frame, 
                                         text="Orientation: [0, 0, 0]",
                                         style='Large.TLabel')
        self.orientation_label.pack(pady=5)
        
        self.velocity_label = ttk.Label(self.values_frame, 
                                      text="Velocity: [0, 0, 0]",
                                      style='Large.TLabel')
        self.velocity_label.pack(pady=5)
        
        # Add simulation speed control
        ttk.Label(self.control_frame, text="Simulation Speed", 
                 style='Large.TLabel').pack(pady=10)
        
        self.speed_slider = ttk.Scale(self.control_frame, 
                                    from_=0.1, to=2.0, 
                                    orient="horizontal",
                                    length=400)
        self.speed_slider.set(1.0)
        self.speed_slider.pack(pady=5)
        
        speed_label = ttk.Label(self.control_frame, 
                               text="1.0x", 
                               style='Large.TLabel')
        speed_label.pack()
        
        def update_speed(event):
            self.sim_speed = self.speed_slider.get()
            speed_label.config(text=f"{self.sim_speed:.1f}x")
        
        self.speed_slider.bind("<Motion>", update_speed)
    
    def create_slider(self, label, min_val, max_val, width=300):
        """Create a labeled slider with value display"""
        frame = ttk.Frame(self.control_frame)
        frame.pack(padx=10, pady=10, fill="x")
        
        ttk.Label(frame, text=label, style='Large.TLabel').pack()
        
        slider = ttk.Scale(frame, from_=min_val, to=max_val, 
                          orient="horizontal", length=width)
        slider.set((min_val + max_val) / 2)
        slider.pack(fill="x", pady=5)
        
        value_label = ttk.Label(frame, text=f"Value: {slider.get():.0f}",
                               style='Large.TLabel')
        value_label.pack()
        
        def update_label(event=None):
            value_label.config(text=f"Value: {slider.get():.0f}")
            self.update_drone()
        
        # Respond to both motion and click events
        slider.bind("<Motion>", update_label)
        slider.bind("<Button-1>", update_label)
        slider.bind("<ButtonRelease-1>", update_label)
        
        return slider
    
    def draw_drone(self, ax, position, orientation):
        """Draw drone body with current position and orientation"""
        # Drone arm length
        L = 0.2
        
        # Create drone body points
        points = np.array([
            [L, 0, 0],    # Front right
            [-L, 0, 0],   # Back right
            [0, L, 0],    # Front left
            [0, -L, 0],   # Back left
        ])
        
        # Apply rotation
        R = self.drone.rotation_matrix(orientation)
        rotated_points = np.dot(points, R.T)
        
        # Apply translation
        translated_points = rotated_points + position
        
        # Draw arms with thicker lines
        ax.plot([position[0], translated_points[0, 0]], 
                [position[1], translated_points[0, 1]], 
                [position[2], translated_points[0, 2]], 'r-', linewidth=2)
        ax.plot([position[0], translated_points[1, 0]], 
                [position[1], translated_points[1, 1]], 
                [position[2], translated_points[1, 2]], 'b-', linewidth=2)
        ax.plot([position[0], translated_points[2, 0]], 
                [position[1], translated_points[2, 1]], 
                [position[2], translated_points[2, 2]], 'g-', linewidth=2)
        ax.plot([position[0], translated_points[3, 0]], 
                [position[1], translated_points[3, 1]], 
                [position[2], translated_points[3, 2]], 'y-', linewidth=2)
        
        # Draw coordinate frame
        axis_length = 0.3
        # Origin frame (fixed)
        if position[0] < 0.1 and position[1] < 0.1 and position[2] < 0.1:  # Only draw at start
            ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', alpha=0.5, label='X')
            ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', alpha=0.5, label='Y')
            ax.quiver(0, 0, 0, 0, 0, axis_length, color='b', alpha=0.5, label='Z')
        
        # Drone's body frame
        axes = np.array([
            [axis_length, 0, 0],  # X-axis
            [0, axis_length, 0],  # Y-axis
            [0, 0, axis_length]   # Z-axis
        ])
        
        # Rotate and translate axes
        rotated_axes = np.dot(axes, R.T)
        
        # Draw rotated coordinate frame
        ax.quiver(position[0], position[1], position[2],
                 rotated_axes[0, 0], rotated_axes[0, 1], rotated_axes[0, 2],
                 color='r', alpha=0.8, linewidth=2)
        ax.quiver(position[0], position[1], position[2],
                 rotated_axes[1, 0], rotated_axes[1, 1], rotated_axes[1, 2],
                 color='g', alpha=0.8, linewidth=2)
        ax.quiver(position[0], position[1], position[2],
                 rotated_axes[2, 0], rotated_axes[2, 1], rotated_axes[2, 2],
                 color='b', alpha=0.8, linewidth=2)
        
        # Add text labels for axes
        offset = 0.1
        ax.text(position[0] + rotated_axes[0, 0] + offset, 
                position[1] + rotated_axes[0, 1] + offset, 
                position[2] + rotated_axes[0, 2] + offset, 
                'X', color='red')
        ax.text(position[0] + rotated_axes[1, 0] + offset, 
                position[1] + rotated_axes[1, 1] + offset, 
                position[2] + rotated_axes[1, 2] + offset, 
                'Y', color='green')
        ax.text(position[0] + rotated_axes[2, 0] + offset, 
                position[1] + rotated_axes[2, 1] + offset, 
                position[2] + rotated_axes[2, 2] + offset, 
                'Z', color='blue')
    
    def start_simulation(self):
        """Start animation"""
        self.is_running = True
    
    def pause_simulation(self):
        """Pause animation"""
        self.is_running = False
    
    def set_hover_speed(self):
        """Set all rotors to hover speed"""
        hover_speed = np.sqrt(self.drone.m * self.drone.g / (4 * self.drone.k))
        for slider in [self.w1_slider, self.w2_slider, self.w3_slider, self.w4_slider]:
            slider.set(hover_speed)
        self.update_drone()
    
    def set_zero_speed(self):
        """Set all rotors to zero speed"""
        for slider in [self.w1_slider, self.w2_slider, self.w3_slider, self.w4_slider]:
            slider.set(0)
        self.update_drone()
        
        # Ensure simulation is running to see the effect
        self.start_simulation()
    
    def update_drone(self):
        """Update drone state based on current slider values"""
        if self.is_running:
            # Update real-time values display
            self.position_label.config(
                text=f"Position: [{self.state[0]:.2f}, {self.state[1]:.2f}, {self.state[2]:.2f}]")
            self.orientation_label.config(
                text=f"Orientation: [{np.rad2deg(self.state[6]):.1f}°, {np.rad2deg(self.state[7]):.1f}°, {np.rad2deg(self.state[8]):.1f}°]")
            self.velocity_label.config(
                text=f"Velocity: [{self.state[3]:.2f}, {self.state[4]:.2f}, {self.state[5]:.2f}]")
            
            # Update canvas
            self.canvas.draw()
    
    def run(self):
        """Main loop"""
        self.root.mainloop()
    
    def create_emergency_visualization(self):
        """Create a new window with flight history visualization"""
        # Create new window
        history_window = tk.Toplevel(self.root)
        history_window.title("Flight History Analysis")
        history_window.geometry("1200x800")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(12, 8))
        
        # 3D trajectory plot
        ax1 = fig.add_subplot(221, projection='3d')
        pos_history = np.array(self.position_history)
        
        # Plot full trajectory
        ax1.plot3D(pos_history[:, 0], pos_history[:, 1], pos_history[:, 2], 
                  'b-', label='Flight Path')
        
        # Mark start and end points
        ax1.scatter(*pos_history[0], color='green', s=100, label='Start')
        ax1.scatter(*pos_history[-1], color='red', s=100, label='Emergency Stop')
        
        # Add ground plane
        x_min, x_max = pos_history[:, 0].min() - 1, pos_history[:, 0].max() + 1
        y_min, y_max = pos_history[:, 1].min() - 1, pos_history[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10),
                            np.linspace(y_min, y_max, 10))
        zz = np.zeros_like(xx)
        ax1.plot_surface(xx, yy, zz, alpha=0.1, color='gray')
        
        ax1.set_title('Complete Flight Path')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.legend()
        
        # Position vs Time plot
        ax2 = fig.add_subplot(222)
        time_points = np.linspace(0, len(pos_history) * self.drone.dt, len(pos_history))
        ax2.plot(time_points, pos_history[:, 0], 'r-', label='X')
        ax2.plot(time_points, pos_history[:, 1], 'g-', label='Y')
        ax2.plot(time_points, pos_history[:, 2], 'b-', label='Z')
        ax2.set_title('Position History')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position (m)')
        ax2.grid(True)
        ax2.legend()
        
        # Orientation history
        ax3 = fig.add_subplot(223)
        orient_history = np.array(self.orientation_history)
        ax3.plot(time_points, np.rad2deg(orient_history[:, 0]), 'r-', label='Roll')
        ax3.plot(time_points, np.rad2deg(orient_history[:, 1]), 'g-', label='Pitch')
        ax3.plot(time_points, np.rad2deg(orient_history[:, 2]), 'b-', label='Yaw')
        ax3.set_title('Orientation History')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Angle (deg)')
        ax3.grid(True)
        ax3.legend()
        
        # Velocity plot (derived from position)
        ax4 = fig.add_subplot(224)
        velocity = np.diff(pos_history, axis=0) / self.drone.dt
        time_points_vel = time_points[:-1]
        ax4.plot(time_points_vel, velocity[:, 0], 'r-', label='Vx')
        ax4.plot(time_points_vel, velocity[:, 1], 'g-', label='Vy')
        ax4.plot(time_points_vel, velocity[:, 2], 'b-', label='Vz')
        ax4.set_title('Velocity History')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Velocity (m/s)')
        ax4.grid(True)
        ax4.legend()
        
        # Add flight statistics
        stats_text = (
            f"Flight Statistics:\n"
            f"Duration: {time_points[-1]:.2f} s\n"
            f"Max Height: {np.max(pos_history[:, 2]):.2f} m\n"
            f"Final Position: [{pos_history[-1, 0]:.2f}, {pos_history[-1, 1]:.2f}, {pos_history[-1, 2]:.2f}]\n"
            f"Max Speed: {np.max(np.linalg.norm(velocity, axis=1)):.2f} m/s"
        )
        fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace')
        
        plt.tight_layout()
        
        # Embed in tkinter window
        canvas = FigureCanvasTkAgg(fig, master=history_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Add save button
        def save_history():
            filename = tk.filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            if filename:
                fig.savefig(filename, dpi=300, bbox_inches='tight')
        
        ttk.Button(history_window, text="Save Plot", 
                  command=save_history).pack(pady=10)
    
    def emergency_stop(self):
        """Handle emergency stop"""
        # Stop motors
        self.set_zero_speed()
        
        # Pause simulation
        self.pause_simulation()
        
        # Show history visualization
        self.create_emergency_visualization()
    
    def create_time_controls(self):
        """Create time control panel"""
        time_frame = ttk.LabelFrame(self.control_frame, text="Time Control")
        time_frame.pack(fill="x", padx=5, pady=5)
        
        # Time multiplier slider
        ttk.Label(time_frame, text="Simulation Speed", 
                 style='Large.TLabel').pack()
        
        self.time_slider = ttk.Scale(time_frame, 
                                   from_=0.1, to=2.0, 
                                   orient="horizontal",
                                   length=300)
        self.time_slider.set(1.0)
        self.time_slider.pack(fill="x", padx=5)
        
        # Time display
        self.time_label = ttk.Label(time_frame, 
                                  text="Real-time (1.0x)", 
                                  style='Large.TLabel')
        self.time_label.pack()
        
        def update_time_multiplier(event=None):
            self.time_multiplier = self.time_slider.get()
            self.time_label.config(text=f"Simulation Speed: {self.time_multiplier:.1f}x")
        
        self.time_slider.configure(command=update_time_multiplier)

if __name__ == "__main__":
    visualizer = DroneDynamicsVisualizer()
    visualizer.run() 