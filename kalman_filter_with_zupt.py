#!/usr/bin/env python3
"""
Kalman Filter with Zero-velocity Update (ZUPT) for IMU data.
Handles stationary periods and reduces drift accumulation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal


class KalmanFilterWithZUPT:
    """
    Kalman Filter with Zero-velocity Update for IMU data.
    
    State vector: [x_pos, y_pos, x_vel, y_vel, x_acc_bias, y_acc_bias]
    Measurement vector: [x_acc, y_acc]
    """
    
    def __init__(self, process_noise_base=1e-4, measurement_noise=1e-2, 
                 zupt_threshold=0.02, window_size=10):
        """
        Initialize Kalman Filter with ZUPT capability.
        
        Parameters:
        -----------
        process_noise_base : float
            Base process noise variance
        measurement_noise : float
            Measurement noise variance
        zupt_threshold : float
            Threshold for zero-velocity detection (m/s²)
        window_size : int
            Window size for moving statistics
        """
        self.process_noise_base = process_noise_base
        self.zupt_threshold = zupt_threshold
        self.window_size = window_size
        
        # State vector: [x_pos, y_pos, x_vel, y_vel, x_acc_bias, y_acc_bias]
        self.state = np.zeros(6)
        
        # Measurement matrix (we measure acceleration minus bias)
        self.H = np.array([
            [0, 0, 0, 0, -1, 0],  # x_acc measurement (with bias)
            [0, 0, 0, 0, 0, -1]   # y_acc measurement (with bias)
        ])
        
        # Measurement noise covariance matrix
        self.R = np.eye(2) * measurement_noise
        
        # Initial state covariance matrix
        self.P = np.eye(6) * 0.1
        self.P[4, 4] = 0.001  # Lower initial uncertainty for bias
        self.P[5, 5] = 0.001
        
        # ZUPT measurement matrix (velocity should be zero)
        self.H_zupt = np.array([
            [0, 0, 1, 0, 0, 0],  # x_vel = 0
            [0, 0, 0, 1, 0, 0]   # y_vel = 0
        ])
        
        # ZUPT measurement noise (very low when we're confident it's stationary)
        self.R_zupt = np.eye(2) * 1e-6
        
        # History for analysis
        self.state_history = []
        self.measurement_history = []
        self.timestamp_history = []
        self.dt_history = []
        self.zupt_history = []
        self.acc_magnitude_history = []
        
        # Moving window for statistics
        self.recent_acc = []
        
    def get_state_transition_matrix(self, dt):
        """
        Generate state transition matrix for given time step.
        Assumes constant velocity model with bias states.
        """
        F = np.array([
            [1, 0, dt, 0, 0, 0],     # x_pos
            [0, 1, 0, dt, 0, 0],     # y_pos
            [0, 0, 1, 0, 0, 0],      # x_vel
            [0, 0, 0, 1, 0, 0],      # y_vel
            [0, 0, 0, 0, 1, 0],      # x_acc_bias
            [0, 0, 0, 0, 0, 1]       # y_acc_bias
        ])
        return F
    
    def get_process_noise_matrix(self, dt):
        """
        Generate process noise covariance matrix.
        """
        Q = np.zeros((6, 6))
        
        # Position noise
        Q[0, 0] = self.process_noise_base * dt**4 / 4
        Q[1, 1] = self.process_noise_base * dt**4 / 4
        
        # Velocity noise
        Q[2, 2] = self.process_noise_base * dt**2
        Q[3, 3] = self.process_noise_base * dt**2
        
        # Bias random walk (very slow changes)
        Q[4, 4] = self.process_noise_base * dt * 0.0001
        Q[5, 5] = self.process_noise_base * dt * 0.0001
        
        return Q
    
    def detect_zero_velocity(self, measurement):
        """
        Detect if the device is stationary based on acceleration magnitude.
        
        Parameters:
        -----------
        measurement : np.array
            Current acceleration measurement [x_acc, y_acc]
            
        Returns:
        --------
        bool
            True if device is likely stationary
        """
        # Add to recent measurements
        self.recent_acc.append(measurement)
        if len(self.recent_acc) > self.window_size:
            self.recent_acc.pop(0)
        
        if len(self.recent_acc) < self.window_size:
            return False
        
        # Calculate statistics over window
        recent = np.array(self.recent_acc)
        acc_std = np.std(recent, axis=0)
        acc_magnitude_std = np.std(np.linalg.norm(recent, axis=1))
        
        # Check if acceleration is stable (low standard deviation)
        is_stationary = (acc_std[0] < self.zupt_threshold and 
                        acc_std[1] < self.zupt_threshold and
                        acc_magnitude_std < self.zupt_threshold)
        
        return is_stationary
    
    def predict(self, dt):
        """
        Prediction step of Kalman filter.
        """
        F = self.get_state_transition_matrix(dt)
        Q = self.get_process_noise_matrix(dt)
        
        # Predict state
        self.state = F @ self.state
        
        # Predict covariance
        self.P = F @ self.P @ F.T + Q
        
    def update_acceleration(self, measurement):
        """
        Update step for acceleration measurement.
        """
        # Acceleration measurement with bias correction
        z = np.array(measurement).reshape(2, 1)
        
        # Expected measurement (negative bias)
        h = self.H @ self.state.reshape(6, 1)
        
        # Innovation
        y = z - h
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state with acceleration (updates velocity)
        # Add acceleration to velocity
        self.state[2] += y[0, 0] * self.dt_current  # x_vel += x_acc * dt
        self.state[3] += y[1, 0] * self.dt_current  # y_vel += y_acc * dt
        
        # Update bias estimates
        self.state = self.state.reshape(6, 1) + K @ y
        self.state = self.state.flatten()
        
        # Update covariance
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P
    
    def update_zupt(self):
        """
        Apply Zero-velocity Update when stationary.
        """
        # Measurement is zero velocity
        z = np.zeros((2, 1))
        
        # Expected velocity from state
        h = self.H_zupt @ self.state.reshape(6, 1)
        
        # Innovation (how much velocity we need to correct)
        y = z - h
        
        # Innovation covariance
        S = self.H_zupt @ self.P @ self.H_zupt.T + self.R_zupt
        
        # Kalman gain
        K = self.P @ self.H_zupt.T @ np.linalg.inv(S)
        
        # Update state (mainly zeros out velocity)
        self.state = self.state.reshape(6, 1) + K @ y
        self.state = self.state.flatten()
        
        # Update covariance
        I = np.eye(6)
        self.P = (I - K @ self.H_zupt) @ self.P
        
    def filter_step(self, measurement, dt, timestamp):
        """
        Complete filter step with ZUPT.
        """
        self.dt_current = dt
        
        # Predict
        self.predict(dt)
        
        # Check for zero velocity
        is_stationary = self.detect_zero_velocity(measurement)
        
        # Update with acceleration
        self.update_acceleration(measurement)
        
        # Apply ZUPT if stationary
        if is_stationary:
            self.update_zupt()
        
        # Store history
        self.state_history.append(self.state.copy())
        self.measurement_history.append(measurement)
        self.timestamp_history.append(timestamp)
        self.dt_history.append(dt)
        self.zupt_history.append(is_stationary)
        self.acc_magnitude_history.append(np.linalg.norm(measurement))
        
    def get_position(self):
        """Get current position estimate."""
        return self.state[0:2]
    
    def get_velocity(self):
        """Get current velocity estimate."""
        return self.state[2:4]
    
    def get_bias(self):
        """Get current bias estimate."""
        return self.state[4:6]


def analyze_stationary_data(data):
    """
    Analyze if data appears to be stationary.
    
    Parameters:
    -----------
    data : pd.DataFrame
        IMU data with x, y acceleration
        
    Returns:
    --------
    dict
        Analysis results
    """
    acc_x = data['x'].values
    acc_y = data['y'].values
    
    # Calculate magnitude
    acc_magnitude = np.sqrt(acc_x**2 + acc_y**2)
    
    # Statistics
    results = {
        'x_std': np.std(acc_x),
        'y_std': np.std(acc_y),
        'magnitude_std': np.std(acc_magnitude),
        'x_range': np.max(acc_x) - np.min(acc_x),
        'y_range': np.max(acc_y) - np.min(acc_y),
        'is_likely_stationary': False
    }
    
    # Check if likely stationary (low standard deviation relative to mean)
    if results['magnitude_std'] < 0.05:  # Less than 0.05 m/s² std
        results['is_likely_stationary'] = True
    
    return results


def process_imu_with_zupt(data, remove_gravity=True):
    """
    Process IMU data with ZUPT-enabled Kalman filter.
    """
    # Get timestamps
    timestamps = data['timestamp'].values
    time_diffs = np.diff(timestamps) / 1000.0  # Convert to seconds
    
    # Analyze data
    analysis = analyze_stationary_data(data)
    print(f"\nData analysis:")
    print(f"  X std dev: {analysis['x_std']:.4f} m/s²")
    print(f"  Y std dev: {analysis['y_std']:.4f} m/s²")
    print(f"  Magnitude std: {analysis['magnitude_std']:.4f} m/s²")
    print(f"  Likely stationary: {analysis['is_likely_stationary']}")
    
    # Get measurements
    acc_x = data['x'].values
    acc_y = data['y'].values
    
    # Remove gravity/bias
    if remove_gravity:
        bias_x = np.mean(acc_x)
        bias_y = np.mean(acc_y)
        acc_x = acc_x - bias_x
        acc_y = acc_y - bias_y
        print(f"\nRemoved bias - X: {bias_x:.4f}, Y: {bias_y:.4f}")
    
    # Calculate measurement noise
    acc_data = np.column_stack([acc_x, acc_y])
    measurement_cov = np.cov(acc_data.T)
    measurement_noise = np.sqrt(np.mean(np.diag(measurement_cov)))
    
    # Adjust ZUPT threshold based on noise level
    zupt_threshold = max(0.01, min(0.1, measurement_noise * 2))
    
    print(f"Measurement noise: {measurement_noise:.6f}")
    print(f"ZUPT threshold: {zupt_threshold:.6f}")
    
    # Initialize filter
    kf = KalmanFilterWithZUPT(
        process_noise_base=1e-5 if analysis['is_likely_stationary'] else 1e-4,
        measurement_noise=measurement_noise,
        zupt_threshold=zupt_threshold,
        window_size=20 if analysis['is_likely_stationary'] else 10
    )
    
    # Set initial bias estimate if stationary
    if analysis['is_likely_stationary']:
        kf.state[4] = 0  # Already removed bias
        kf.state[5] = 0
    
    # Process measurements
    print(f"\nProcessing {len(data)} measurements with ZUPT...")
    
    # First measurement
    measurement = [acc_x[0], acc_y[0]]
    kf.filter_step(measurement, np.median(time_diffs), timestamps[0])
    
    # Remaining measurements
    for i in range(1, len(data)):
        dt = (timestamps[i] - timestamps[i-1]) / 1000.0
        
        if dt > 0.1:
            dt = 0.1
        elif dt < 0.001:
            continue
            
        measurement = [acc_x[i], acc_y[i]]
        kf.filter_step(measurement, dt, timestamps[i])
        
        if (i + 1) % 5000 == 0:
            zupt_count = sum(kf.zupt_history)
            print(f"  Processed {i + 1}/{len(data)} samples (ZUPT applied: {zupt_count} times)")
    
    zupt_count = sum(kf.zupt_history)
    zupt_percentage = (zupt_count / len(kf.zupt_history)) * 100
    print(f"\nZUPT applied {zupt_count} times ({zupt_percentage:.1f}% of samples)")
    
    return kf


def plot_zupt_results(kf, save_path='kalman_zupt_results.png'):
    """
    Plot results with ZUPT analysis.
    """
    history = np.array(kf.state_history)
    timestamps = np.array(kf.timestamp_history)
    zupt = np.array(kf.zupt_history)
    acc_mag = np.array(kf.acc_magnitude_history)
    
    time = (timestamps - timestamps[0]) / 1000.0
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Position trajectory
    ax = axes[0, 0]
    ax.plot(history[:, 0], history[:, 1], 'b-', alpha=0.7, label='Trajectory')
    ax.plot(history[0, 0], history[0, 1], 'go', markersize=10, label='Start')
    ax.plot(history[-1, 0], history[-1, 1], 'ro', markersize=10, label='End')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Position Trajectory with ZUPT')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 2: ZUPT activation
    ax = axes[0, 1]
    zupt_indices = np.where(zupt)[0]
    ax.plot(time, acc_mag, 'b-', alpha=0.5, label='Acc magnitude')
    if len(zupt_indices) > 0:
        ax.scatter(time[zupt_indices], acc_mag[zupt_indices], c='r', s=1, label='ZUPT active')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration Magnitude (m/s²)')
    ax.set_title('ZUPT Detection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Velocity
    ax = axes[0, 2]
    ax.plot(time, history[:, 2], 'b-', alpha=0.7, label='X velocity')
    ax.plot(time, history[:, 3], 'r-', alpha=0.7, label='Y velocity')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity with ZUPT Corrections')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Position over time
    ax = axes[1, 0]
    ax.plot(time, history[:, 0], 'b-', label='X position')
    ax.plot(time, history[:, 1], 'r-', label='Y position')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m)')
    ax.set_title('Position Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Bias estimates
    ax = axes[1, 1]
    ax.plot(time, history[:, 4], 'b-', label='X bias')
    ax.plot(time, history[:, 5], 'r-', label='Y bias')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Bias (m/s²)')
    ax.set_title('Estimated Acceleration Bias')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Distance from origin
    ax = axes[1, 2]
    distances = np.linalg.norm(history[:, 0:2], axis=1)
    ax.plot(time, distances, 'g-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance (m)')
    ax.set_title('Distance from Origin')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Kalman Filter with ZUPT - IMU Data', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"Plot saved to {save_path}")


def main():
    """Main function."""
    from read_imu_data import read_imu_data
    
    print("=" * 60)
    print("KALMAN FILTER WITH ZUPT")
    print("=" * 60)
    
    # Process statistics file (stationary data)
    print("\nProcessing STATISTICS file (stationary/noise data)...")
    stats_data = read_imu_data('data/statistics.txt')
    print(f"Loaded {len(stats_data)} samples")
    
    kf_stats = process_imu_with_zupt(stats_data, remove_gravity=True)
    
    # Calculate final position
    history = np.array(kf_stats.state_history)
    end_pos = history[-1, 0:2]
    final_distance = np.linalg.norm(end_pos)
    
    print(f"\nResults for stationary data:")
    print(f"  Final position: ({end_pos[0]:.4f}, {end_pos[1]:.4f}) m")
    print(f"  Distance from origin: {final_distance:.4f} m")
    print(f"  Expected: ~0 m (stationary)")
    print(f"  Drift rate: {final_distance / (len(stats_data) * 0.008):.6f} m/s")
    
    plot_zupt_results(kf_stats, 'kalman_zupt_statistics.png')
    
    # Process data1 (with movement)
    print("\n" + "=" * 60)
    print("Processing DATA1 file (with movement)...")
    data1 = read_imu_data('data/data1.txt')
    print(f"Loaded {len(data1)} samples")
    
    kf_data1 = process_imu_with_zupt(data1, remove_gravity=True)
    
    history = np.array(kf_data1.state_history)
    end_pos = history[-1, 0:2]
    final_distance = np.linalg.norm(end_pos)
    
    print(f"\nResults for movement data:")
    print(f"  Final position: ({end_pos[0]:.4f}, {end_pos[1]:.4f}) m")
    print(f"  Distance traveled: {final_distance:.4f} m")
    
    plot_zupt_results(kf_data1, 'kalman_zupt_data1.png')


if __name__ == "__main__":
    main()