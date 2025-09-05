#!/usr/bin/env python3
"""
Kalman Filter implementation with variable time steps for IMU data processing.
Handles non-uniform timestamps by updating the state transition matrix at each step.
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


class KalmanFilterVariableDt:
    """
    Kalman Filter for IMU acceleration data with variable time steps.
    
    State vector: [x_pos, y_pos, x_vel, y_vel, x_acc, y_acc]
    Measurement vector: [x_acc, y_acc]
    """
    
    def __init__(self, process_noise_base=1e-3, measurement_noise=1e-2):
        """
        Initialize Kalman Filter for variable time steps.
        
        Parameters:
        -----------
        process_noise_base : float
            Base process noise variance (will be scaled by dt)
        measurement_noise : float
            Measurement noise variance
        """
        self.process_noise_base = process_noise_base
        
        # State vector: [x_pos, y_pos, x_vel, y_vel, x_acc, y_acc]
        self.state = np.zeros(6)
        
        # Measurement matrix (we only measure acceleration)
        self.H = np.array([
            [0, 0, 0, 0, 1, 0],  # x_acc measurement
            [0, 0, 0, 0, 0, 1]   # y_acc measurement
        ])
        
        # Measurement noise covariance matrix
        self.R = np.eye(2) * measurement_noise
        
        # Initial state covariance matrix
        self.P = np.eye(6) * 0.1
        
        # Store history for analysis
        self.state_history = []
        self.measurement_history = []
        self.timestamp_history = []
        self.dt_history = []
        
    def get_state_transition_matrix(self, dt):
        """
        Generate state transition matrix for given time step.
        
        Parameters:
        -----------
        dt : float
            Time step in seconds
            
        Returns:
        --------
        np.array
            6x6 state transition matrix
        """
        F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],          # x_pos
            [0, 1, 0, dt, 0, 0.5*dt**2],          # y_pos
            [0, 0, 1, 0, dt, 0],                  # x_vel
            [0, 0, 0, 1, 0, dt],                  # y_vel
            [0, 0, 0, 0, 1, 0],                   # x_acc
            [0, 0, 0, 0, 0, 1]                    # y_acc
        ])
        return F
    
    def get_process_noise_matrix(self, dt):
        """
        Generate process noise covariance matrix scaled by time step.
        
        Parameters:
        -----------
        dt : float
            Time step in seconds
            
        Returns:
        --------
        np.array
            6x6 process noise covariance matrix
        """
        # Scale noise by dt for position and velocity
        # Higher order terms scale with higher powers of dt
        Q = np.zeros((6, 6))
        
        # Position noise (scales with dt^5)
        Q[0, 0] = self.process_noise_base * dt**5 / 20
        Q[1, 1] = self.process_noise_base * dt**5 / 20
        
        # Velocity noise (scales with dt^3)
        Q[2, 2] = self.process_noise_base * dt**3 / 3
        Q[3, 3] = self.process_noise_base * dt**3 / 3
        
        # Acceleration noise (scales with dt)
        Q[4, 4] = self.process_noise_base * dt * 10
        Q[5, 5] = self.process_noise_base * dt * 10
        
        return Q
        
    def predict(self, dt):
        """
        Prediction step of Kalman filter with variable time step.
        
        Parameters:
        -----------
        dt : float
            Time step since last measurement in seconds
        """
        # Get state transition matrix for this time step
        F = self.get_state_transition_matrix(dt)
        
        # Get process noise for this time step
        Q = self.get_process_noise_matrix(dt)
        
        # Predict state
        self.state = F @ self.state
        
        # Predict covariance
        self.P = F @ self.P @ F.T + Q
        
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
        
    def filter_step(self, measurement, dt, timestamp):
        """
        Complete filter step with variable time step.
        
        Parameters:
        -----------
        measurement : np.array
            Measured acceleration [x_acc, y_acc]
        dt : float
            Time step since last measurement in seconds
        timestamp : float
            Current timestamp
        """
        self.predict(dt)
        self.update(measurement)
        
        # Store history
        self.state_history.append(self.state.copy())
        self.measurement_history.append(measurement)
        self.timestamp_history.append(timestamp)
        self.dt_history.append(dt)
        
    def get_position(self):
        """Get current position estimate."""
        return self.state[0:2]
    
    def get_velocity(self):
        """Get current velocity estimate."""
        return self.state[2:4]
    
    def get_acceleration(self):
        """Get current acceleration estimate."""
        return self.state[4:6]


def process_imu_data_with_variable_dt(data, remove_gravity=True):
    """
    Process IMU data through Kalman filter with variable time steps.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with columns ['timestamp', 'x', 'y']
    remove_gravity : bool
        Whether to remove gravity component (assumed to be in mean)
        
    Returns:
    --------
    KalmanFilterVariableDt
        Fitted Kalman filter with state history
    """
    # Get timestamps and convert to seconds
    timestamps = data['timestamp'].values
    time_diffs = np.diff(timestamps) / 1000.0  # Convert milliseconds to seconds
    
    # Analyze time differences
    print(f"Time step statistics:")
    print(f"  Mean dt: {np.mean(time_diffs):.4f} seconds")
    print(f"  Median dt: {np.median(time_diffs):.4f} seconds")
    print(f"  Std dev: {np.std(time_diffs):.4f} seconds")
    print(f"  Min dt: {np.min(time_diffs):.4f} seconds")
    print(f"  Max dt: {np.max(time_diffs):.4f} seconds")
    
    # Get acceleration measurements
    acc_x = data['x'].values
    acc_y = data['y'].values
    
    # Remove gravity/bias if requested
    if remove_gravity:
        acc_x, acc_y, bias_x, bias_y = remove_gravity_bias(acc_x, acc_y)
        print(f"\nRemoved bias - X: {bias_x:.4f}, Y: {bias_y:.4f}")
    
    # Calculate measurement noise from data
    measurement_noise = estimate_measurement_noise(acc_x, acc_y)
    
    print(f"Estimated measurement noise: {measurement_noise:.6f}")
    
    # Initialize Kalman filter
    kf = KalmanFilterVariableDt(process_noise_base=1e-4, measurement_noise=measurement_noise)
    
    # Process first measurement with small dt
    print(f"\nProcessing {len(data)} measurements with variable time steps...")
    measurement = [acc_x[0], acc_y[0]]
    median_dt = np.median(time_diffs)
    kf.filter_step(measurement, median_dt, timestamps[0])  # Use median dt for first measurement
    
    # Process remaining measurements with actual time differences
    for i in range(1, len(data)):
        dt = (timestamps[i] - timestamps[i-1]) / 1000.0  # Convert milliseconds to seconds
        
        # Skip if dt is too large (possible data gap) or too small
        if dt > 0.1:  # More than 100ms gap
            print(f"  Warning: Large time gap of {dt*1000:.2f}ms at sample {i}, capping at 100ms")
            dt = 0.1
        elif dt < 0.000001:  # Less than 1 microsecond
            print(f"  Warning: Very small dt of {dt*1e6:.2f}μs at sample {i}, skipping")
            continue
            
        measurement = [acc_x[i], acc_y[i]]
        kf.filter_step(measurement, dt, timestamps[i])
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(data)} samples")
    
    return kf




def plot_variable_dt_results(kf, original_data, save_path='kalman_variable_dt_results.png'):
    """
    Plot Kalman filter results with variable dt analysis.
    
    Parameters:
    -----------
    kf : KalmanFilterVariableDt
        Fitted Kalman filter
    original_data : pd.DataFrame
        Original IMU data
    save_path : str
        Path to save the plot
    """
    history = np.array(kf.state_history)
    measurements = np.array(kf.measurement_history)
    timestamps = np.array(kf.timestamp_history)
    dts = np.array(kf.dt_history)
    
    # Convert timestamps to seconds from start
    time = (timestamps - timestamps[0]) / 1000.0
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Position trajectory
    plot_trajectory(history, axes[0, 0])
    
    # Plot 2: Time step variations
    ax = axes[0, 1]
    ax.plot(time[1:], dts[1:]*1000, 'g-', alpha=0.7, linewidth=0.5)
    ax.axhline(y=np.median(dts[1:])*1000, color='r', linestyle='--', label=f'Median: {np.median(dts[1:])*1000:.1f}ms')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Time Step (ms)')
    ax.set_title('Variable Time Steps Between Measurements')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
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
    
    # Plot 5: Velocity over time
    plot_velocity(history, time, axes[1, 1])
    axes[1, 1].set_title('Velocity Estimates (Variable dt)')
    
    # Plot 6: Time step histogram
    ax = axes[1, 2]
    ax.hist(dts[1:]*1000, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Time Step (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Time Steps')
    ax.axvline(x=np.median(dts[1:])*1000, color='r', linestyle='--', label=f'Median: {np.median(dts[1:])*1000:.1f}ms')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Kalman Filter with Variable Time Steps - IMU Data', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"Plot saved to {save_path}")


def main():
    """Main function to demonstrate variable dt Kalman filtering."""
    
    # Read IMU data
    print("=" * 60)
    print("KALMAN FILTER WITH VARIABLE TIME STEPS")
    print("=" * 60)
    
    # Import the read function from our previous script
    from read_imu_data import read_imu_data
    
    # Process data1.txt as an example
    data_file = 'data/data1.txt'
    print(f"\nReading data from {data_file}...")
    imu_data = read_imu_data(data_file)
    print(f"Loaded {len(imu_data)} samples")
    
    # Apply Kalman filter with variable dt
    print("\n" + "=" * 60)
    print("APPLYING KALMAN FILTER (VARIABLE DT)")
    print("=" * 60)
    kf = process_imu_data_with_variable_dt(imu_data, remove_gravity=True)
    
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
    plot_variable_dt_results(kf, imu_data, 'kalman_variable_dt_data1.png')
    
    # Process statistics.txt with more data
    print("\n" + "=" * 60)
    print("PROCESSING STATISTICS FILE")
    print("=" * 60)
    
    stats_data = read_imu_data('data/statistics.txt')
    print(f"Loaded {len(stats_data)} samples from statistics.txt")
    
    kf_stats = process_imu_data_with_variable_dt(stats_data, remove_gravity=True)
    distances_stats = calculate_distance_traveled(kf_stats.state_history)
    
    print(f"\nStatistics file results:")
    print(f"Start position: ({distances_stats['start_position'][0]:.4f}, {distances_stats['start_position'][1]:.4f}) m")
    print(f"End position: ({distances_stats['end_position'][0]:.4f}, {distances_stats['end_position'][1]:.4f}) m")
    print(f"Euclidean distance: {distances_stats['euclidean_distance']:.4f} m")
    print(f"Total path length: {distances_stats['total_path_length']:.4f} m")
    
    plot_variable_dt_results(kf_stats, stats_data, 'kalman_variable_dt_statistics.png')
    
    # Compare with constant dt version
    print("\n" + "=" * 60)
    print("COMPARISON: VARIABLE DT vs CONSTANT DT")
    print("=" * 60)
    print("(Run the previous kalman_filter_imu.py to compare)")


if __name__ == "__main__":
    main()