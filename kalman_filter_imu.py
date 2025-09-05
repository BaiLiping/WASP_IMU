#!/usr/bin/env python3
"""
Kalman Filter implementation for IMU data processing.
Filters acceleration data and estimates position/velocity.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from kalman_base import (
    calculate_distance_traveled,
    plot_trajectory,
    plot_velocity,
    plot_acceleration,
    remove_gravity_bias,
    estimate_measurement_noise
)


class KalmanFilter:
    """
    Kalman Filter for IMU acceleration data.
    
    State vector: [x_pos, y_pos, x_vel, y_vel, x_acc, y_acc]
    Measurement vector: [x_acc, y_acc]
    """
    
    def __init__(self, dt=0.01, process_noise=1e-3, measurement_noise=1e-2):
        """
        Initialize Kalman Filter.
        
        Parameters:
        -----------
        dt : float
            Time step between measurements
        process_noise : float
            Process noise variance
        measurement_noise : float
            Measurement noise variance
        """
        self.dt = dt
        
        # State vector: [x_pos, y_pos, x_vel, y_vel, x_acc, y_acc]
        self.state = np.zeros(6)
        
        # State transition matrix (constant acceleration model)
        self.F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],          # x_pos
            [0, 1, 0, dt, 0, 0.5*dt**2],          # y_pos
            [0, 0, 1, 0, dt, 0],                  # x_vel
            [0, 0, 0, 1, 0, dt],                  # y_vel
            [0, 0, 0, 0, 1, 0],                   # x_acc
            [0, 0, 0, 0, 0, 1]                    # y_acc
        ])
        
        # Measurement matrix (we only measure acceleration)
        self.H = np.array([
            [0, 0, 0, 0, 1, 0],  # x_acc measurement
            [0, 0, 0, 0, 0, 1]   # y_acc measurement
        ])
        
        # Process noise covariance matrix
        self.Q = np.eye(6) * process_noise
        self.Q[4, 4] = process_noise * 10  # Higher noise for acceleration
        self.Q[5, 5] = process_noise * 10
        
        # Measurement noise covariance matrix
        self.R = np.eye(2) * measurement_noise
        
        # Initial state covariance matrix
        self.P = np.eye(6) * 0.1
        
        # Store history for analysis
        self.state_history = []
        self.measurement_history = []
        
    def predict(self):
        """
        Prediction step of Kalman filter.
        Updates state and covariance based on motion model.
        """
        # Predict state
        self.state = self.F @ self.state
        
        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, measurement):
        """
        Update step of Kalman filter.
        
        Parameters:
        -----------
        measurement : np.array
            Measured acceleration [x_acc, y_acc]
        """
        # Convert measurement to numpy array if needed
        z = np.array(measurement).reshape(2, 1)
        
        # Innovation (measurement residual)
        y = z - self.H @ self.state.reshape(6, 1)
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state.reshape(6, 1) + K @ y
        self.state = self.state.flatten()
        
        # Update covariance
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P
        
        # Store history
        self.state_history.append(self.state.copy())
        self.measurement_history.append(measurement)
        
    def filter_step(self, measurement):
        """
        Complete filter step: predict then update.
        
        Parameters:
        -----------
        measurement : np.array
            Measured acceleration [x_acc, y_acc]
        """
        self.predict()
        self.update(measurement)
        
    def get_position(self):
        """Get current position estimate."""
        return self.state[0:2]
    
    def get_velocity(self):
        """Get current velocity estimate."""
        return self.state[2:4]
    
    def get_acceleration(self):
        """Get current acceleration estimate."""
        return self.state[4:6]


def process_imu_data_with_kalman(data, dt=None, remove_gravity=True):
    """
    Process IMU data through Kalman filter.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with columns ['timestamp', 'x', 'y']
    dt : float, optional
        Time step. If None, computed from timestamps
    remove_gravity : bool
        Whether to remove gravity component (assumed to be in mean)
        
    Returns:
    --------
    KalmanFilter
        Fitted Kalman filter with state history
    """
    # Calculate time step if not provided
    if dt is None:
        # Convert timestamps to seconds
        timestamps = data['timestamp'].values
        time_diffs = np.diff(timestamps) / 1000.0  # Convert ms to seconds
        dt = np.median(time_diffs)  # Use median to avoid outliers
    
    print(f"Using time step dt = {dt:.4f} seconds")
    
    # Get acceleration measurements
    acc_x = data['x'].values
    acc_y = data['y'].values
    
    # Remove gravity/bias if requested
    if remove_gravity:
        acc_x, acc_y, bias_x, bias_y = remove_gravity_bias(acc_x, acc_y)
        print(f"Removed bias - X: {bias_x:.4f}, Y: {bias_y:.4f}")
    
    # Calculate measurement noise from data
    measurement_noise = estimate_measurement_noise(acc_x, acc_y)
    
    print(f"Estimated measurement noise: {measurement_noise:.6f}")
    
    # Initialize Kalman filter
    kf = KalmanFilter(dt=dt, process_noise=1e-4, measurement_noise=measurement_noise)
    
    # Process all measurements
    print(f"Processing {len(data)} measurements...")
    for i in range(len(data)):
        measurement = [acc_x[i], acc_y[i]]
        kf.filter_step(measurement)
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(data)} samples")
    
    return kf




def plot_kalman_results(kf, original_data, save_path='kalman_results.png'):
    """
    Plot Kalman filter results.
    
    Parameters:
    -----------
    kf : KalmanFilter
        Fitted Kalman filter
    original_data : pd.DataFrame
        Original IMU data
    save_path : str
        Path to save the plot
    """
    history = np.array(kf.state_history)
    measurements = np.array(kf.measurement_history)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Position trajectory
    plot_trajectory(history, axes[0, 0])
    
    # Plot 2: Velocity over time
    time = np.arange(len(history)) * kf.dt
    plot_velocity(history, time, axes[0, 1])
    
    # Plot 3: Acceleration (filtered vs measured)
    plot_acceleration(history, measurements, time, axes[0, 2])
    
    # Plot 4: Position over time
    ax = axes[1, 0]
    ax.plot(time, history[:, 0], 'b-', label='X position')
    ax.plot(time, history[:, 1], 'r-', label='Y position')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m)')
    ax.set_title('Position Components over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Distance from origin
    ax = axes[1, 1]
    distances = np.linalg.norm(history[:, 0:2], axis=1)
    ax.plot(time, distances, 'g-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance from Origin (m)')
    ax.set_title('Distance from Starting Point')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Speed magnitude
    ax = axes[1, 2]
    speeds = np.linalg.norm(history[:, 2:4], axis=1)
    ax.plot(time, speeds, 'm-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Speed Magnitude')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Kalman Filter Results for IMU Data', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"Plot saved to {save_path}")


def main():
    """Main function to demonstrate Kalman filtering on IMU data."""
    
    # Read IMU data
    print("=" * 60)
    print("KALMAN FILTER FOR IMU DATA")
    print("=" * 60)
    
    # Import the read function from our previous script
    from read_imu_data import read_imu_data
    
    # Process data1.txt as an example
    data_file = 'data/data1.txt'
    print(f"\nReading data from {data_file}...")
    imu_data = read_imu_data(data_file)
    print(f"Loaded {len(imu_data)} samples")
    
    # Apply Kalman filter
    print("\n" + "=" * 60)
    print("APPLYING KALMAN FILTER")
    print("=" * 60)
    kf = process_imu_data_with_kalman(imu_data, remove_gravity=True)
    
    # Calculate distances
    print("\n" + "=" * 60)
    print("DISTANCE CALCULATIONS")
    print("=" * 60)
    distances = calculate_distance_traveled(kf.state_history)
    
    print(f"\nStart position: ({distances['start_position'][0]:.4f}, {distances['start_position'][1]:.4f}) m")
    print(f"End position: ({distances['end_position'][0]:.4f}, {distances['end_position'][1]:.4f}) m")
    print(f"Displacement vector: ({distances['displacement_vector'][0]:.4f}, {distances['displacement_vector'][1]:.4f}) m")
    print(f"\nEuclidean distance (start to end): {distances['euclidean_distance']:.4f} m")
    print(f"Total path length traveled: {distances['total_path_length']:.4f} m")
    print(f"Maximum displacement from origin: {distances['max_displacement']:.4f} m")
    
    # Plot results
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    plot_kalman_results(kf, imu_data, 'kalman_filter_results.png')
    
    # Process statistics.txt with more data
    print("\n" + "=" * 60)
    print("PROCESSING STATISTICS FILE")
    print("=" * 60)
    
    stats_data = read_imu_data('data/statistics.txt')
    print(f"Loaded {len(stats_data)} samples from statistics.txt")
    
    kf_stats = process_imu_data_with_kalman(stats_data, remove_gravity=True)
    distances_stats = calculate_distance_traveled(kf_stats.state_history)
    
    print(f"\nStatistics file results:")
    print(f"Euclidean distance: {distances_stats['euclidean_distance']:.4f} m")
    print(f"Total path length: {distances_stats['total_path_length']:.4f} m")
    
    plot_kalman_results(kf_stats, stats_data, 'kalman_filter_statistics.png')


if __name__ == "__main__":
    main()