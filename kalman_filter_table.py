#!/usr/bin/env python3
"""
Kalman Filter for table_data.txt with constant velocity model.
Uses measured noise statistics from the stationary data analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class KalmanFilterConstantVelocity:
    """
    Kalman Filter for IMU data with constant velocity motion model.
    Incorporates measured noise statistics from sensor characterization.
    
    State vector: [x_pos, y_pos, x_vel, y_vel]
    Measurement vector: [x_acc, y_acc]
    """
    
    def __init__(self, noise_stats=None):
        """
        Initialize Kalman Filter with measured noise statistics.
        
        Parameters:
        -----------
        noise_stats : dict
            Dictionary containing measured noise parameters
        """
        # State vector: [x_pos, y_pos, x_vel, y_vel]
        self.state = np.zeros(4)
        
        # Use measured noise statistics if provided
        if noise_stats:
            # White noise standard deviation from measurements
            self.sigma_acc_x = noise_stats.get('sigma_x', 0.0293)  # m/s²
            self.sigma_acc_y = noise_stats.get('sigma_y', 0.0113)  # m/s²
            
            # Bias values
            self.bias_x = noise_stats.get('bias_x', 0.0664)
            self.bias_y = noise_stats.get('bias_y', 0.0525)
        else:
            # Default values from our noise analysis
            self.sigma_acc_x = 0.0293  # m/s²
            self.sigma_acc_y = 0.0113  # m/s²
            self.bias_x = 0.0664
            self.bias_y = 0.0525
        
        # Initial state covariance matrix
        self.P = np.eye(4)
        self.P[0, 0] = 0.01  # Position uncertainty (m²)
        self.P[1, 1] = 0.01
        self.P[2, 2] = 0.001  # Velocity uncertainty (m²/s²)
        self.P[3, 3] = 0.001
        
        # History for analysis
        self.state_history = []
        self.measurement_history = []
        self.timestamp_history = []
        self.dt_history = []
        self.innovation_history = []
        
    def get_state_transition_matrix(self, dt):
        """
        State transition matrix for constant velocity model.
        
        Parameters:
        -----------
        dt : float
            Time step in seconds
            
        Returns:
        --------
        F : np.array
            4x4 state transition matrix
        """
        F = np.array([
            [1, 0, dt, 0],   # x_pos = x_pos + x_vel * dt
            [0, 1, 0, dt],   # y_pos = y_pos + y_vel * dt
            [0, 0, 1, 0],    # x_vel = x_vel (constant velocity)
            [0, 0, 0, 1]     # y_vel = y_vel (constant velocity)
        ])
        return F
    
    def get_process_noise_matrix(self, dt):
        """
        Process noise covariance matrix.
        Accounts for unmodeled accelerations and velocity changes.
        
        Parameters:
        -----------
        dt : float
            Time step in seconds
            
        Returns:
        --------
        Q : np.array
            4x4 process noise covariance matrix
        """
        # Process noise for velocity (using measured acceleration noise)
        # This represents how much the velocity can change due to unmodeled accelerations
        q_vel_x = (self.sigma_acc_x * dt) ** 2
        q_vel_y = (self.sigma_acc_y * dt) ** 2
        
        # Process noise for position (integrated velocity uncertainty)
        q_pos_x = (self.sigma_acc_x * dt**2 / 2) ** 2
        q_pos_y = (self.sigma_acc_y * dt**2 / 2) ** 2
        
        Q = np.array([
            [q_pos_x, 0, q_pos_x/dt, 0],
            [0, q_pos_y, 0, q_pos_y/dt],
            [q_pos_x/dt, 0, q_vel_x, 0],
            [0, q_pos_y/dt, 0, q_vel_y]
        ])
        
        return Q
    
    def get_measurement_matrix(self, dt):
        """
        Measurement matrix relating acceleration to state.
        For constant velocity model, measured acceleration should be ~0.
        
        Parameters:
        -----------
        dt : float
            Time step in seconds
            
        Returns:
        --------
        H : np.array
            2x4 measurement matrix
        """
        # We measure acceleration, which is the derivative of velocity
        # For constant velocity model, we expect zero acceleration
        # So we measure the change in velocity divided by dt
        H = np.array([
            [0, 0, 0, 0],  # x_acc ≈ 0 for constant velocity
            [0, 0, 0, 0]   # y_acc ≈ 0 for constant velocity
        ])
        # Note: This will be handled differently in the update step
        return H
    
    def get_measurement_noise_matrix(self):
        """
        Measurement noise covariance matrix using measured noise statistics.
        
        Returns:
        --------
        R : np.array
            2x2 measurement noise covariance matrix
        """
        # Use measured white noise variances
        R = np.array([
            [self.sigma_acc_x ** 2, 0],
            [0, self.sigma_acc_y ** 2]
        ])
        return R
    
    def predict(self, dt):
        """
        Prediction step of Kalman filter.
        
        Parameters:
        -----------
        dt : float
            Time step since last measurement in seconds
        """
        # Get state transition matrix
        F = self.get_state_transition_matrix(dt)
        
        # Get process noise matrix
        Q = self.get_process_noise_matrix(dt)
        
        # Predict state
        self.state = F @ self.state
        
        # Predict covariance
        self.P = F @ self.P @ F.T + Q
        
    def update(self, measurement, dt):
        """
        Update step for acceleration measurement.
        
        Parameters:
        -----------
        measurement : np.array
            Measured acceleration [x_acc, y_acc] in m/s²
        dt : float
            Time step in seconds
        """
        # For constant velocity model, we use acceleration to update velocity
        # The innovation is the difference between measured and expected acceleration
        
        # Expected acceleration for constant velocity is zero
        expected_acc = np.array([0.0, 0.0])
        
        # Innovation (measurement residual)
        innovation = measurement - expected_acc
        self.innovation_history.append(innovation)
        
        # Measurement matrix for velocity update from acceleration
        # H maps acceleration to velocity change
        H = np.array([
            [0, 0, 1/dt, 0],  # x_acc affects x_vel
            [0, 0, 0, 1/dt]   # y_acc affects y_vel
        ])
        
        # Measurement noise
        R = self.get_measurement_noise_matrix()
        
        # Innovation covariance
        S = H @ self.P @ H.T + R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        # Use acceleration to update velocity
        velocity_update = innovation * dt  # Convert acceleration to velocity change
        state_update = K @ velocity_update
        self.state = self.state + state_update
        
        # Update covariance
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P
        
    def filter_step(self, measurement, dt, timestamp):
        """
        Complete filter step: predict then update.
        
        Parameters:
        -----------
        measurement : np.array
            Measured acceleration [x_acc, y_acc]
        dt : float
            Time step since last measurement in seconds
        timestamp : float
            Current timestamp
        """
        # Predict
        self.predict(dt)
        
        # Update
        self.update(measurement, dt)
        
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


def process_table_data(data_path='data/table_data.txt'):
    """
    Process table_data.txt with Kalman filter using measured noise statistics.
    
    Parameters:
    -----------
    data_path : str
        Path to the table data file
        
    Returns:
    --------
    kf : KalmanFilterConstantVelocity
        Fitted Kalman filter
    data : pd.DataFrame
        Original data
    """
    # Read data
    print("Reading table_data.txt...")
    from read_imu_data import read_imu_data
    data = read_imu_data(data_path)
    print(f"Loaded {len(data)} samples")
    
    # Get timestamps and calculate dt
    timestamps = data['timestamp'].values
    time_diffs = np.diff(timestamps) / 1000.0  # Convert ms to seconds
    
    print(f"\nTime step statistics:")
    print(f"  Mean dt: {np.mean(time_diffs)*1000:.2f} ms")
    print(f"  Median dt: {np.median(time_diffs)*1000:.2f} ms")
    print(f"  Std dev: {np.std(time_diffs)*1000:.2f} ms")
    
    # Get acceleration measurements
    acc_x = data['x'].values
    acc_y = data['y'].values
    
    # Calculate statistics from this data
    print(f"\nRaw data statistics:")
    print(f"  X mean: {np.mean(acc_x):.4f} m/s²")
    print(f"  Y mean: {np.mean(acc_y):.4f} m/s²")
    print(f"  X std: {np.std(acc_x):.4f} m/s²")
    print(f"  Y std: {np.std(acc_y):.4f} m/s²")
    
    # Remove bias (use statistics file values)
    bias_x = 0.0664  # From statistics file analysis
    bias_y = 0.0525  # From statistics file analysis
    
    acc_x_corrected = acc_x - bias_x
    acc_y_corrected = acc_y - bias_y
    
    print(f"\nBias correction applied:")
    print(f"  X bias: {bias_x:.4f} m/s²")
    print(f"  Y bias: {bias_y:.4f} m/s²")
    
    # Create noise statistics dictionary
    noise_stats = {
        'sigma_x': 0.0293,  # From statistics file white noise analysis
        'sigma_y': 0.0113,  # From statistics file white noise analysis
        'bias_x': bias_x,
        'bias_y': bias_y
    }
    
    # Initialize Kalman filter with measured noise statistics
    print(f"\nInitializing Kalman filter with measured noise statistics:")
    print(f"  X-axis white noise σ: {noise_stats['sigma_x']:.4f} m/s²")
    print(f"  Y-axis white noise σ: {noise_stats['sigma_y']:.4f} m/s²")
    
    kf = KalmanFilterConstantVelocity(noise_stats)
    
    # Process measurements
    print(f"\nProcessing {len(data)} measurements...")
    
    # First measurement
    measurement = [acc_x_corrected[0], acc_y_corrected[0]]
    median_dt = np.median(time_diffs)
    kf.filter_step(measurement, median_dt, timestamps[0])
    
    # Process remaining measurements
    for i in range(1, len(data)):
        dt = (timestamps[i] - timestamps[i-1]) / 1000.0  # Convert to seconds
        
        # Skip if dt is too large or too small
        if dt > 0.1:  # More than 100ms
            dt = 0.1
        elif dt < 0.001:  # Less than 1ms
            continue
            
        measurement = [acc_x_corrected[i], acc_y_corrected[i]]
        kf.filter_step(measurement, dt, timestamps[i])
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(data)} samples")
    
    return kf, data


def analyze_results(kf):
    """
    Analyze Kalman filter results.
    
    Parameters:
    -----------
    kf : KalmanFilterConstantVelocity
        Fitted Kalman filter
        
    Returns:
    --------
    dict
        Analysis results
    """
    history = np.array(kf.state_history)
    
    # Extract positions and velocities
    positions = history[:, 0:2]
    velocities = history[:, 2:4]
    
    # Calculate trajectory metrics
    start_pos = positions[0]
    end_pos = positions[-1]
    displacement = end_pos - start_pos
    distance = np.linalg.norm(displacement)
    
    # Calculate path length
    path_segments = np.diff(positions, axis=0)
    path_lengths = np.linalg.norm(path_segments, axis=1)
    total_path = np.sum(path_lengths)
    
    # Velocity statistics
    speed = np.linalg.norm(velocities, axis=1)
    
    results = {
        'start_position': start_pos,
        'end_position': end_pos,
        'displacement': displacement,
        'distance': distance,
        'total_path': total_path,
        'mean_velocity_x': np.mean(velocities[:, 0]),
        'mean_velocity_y': np.mean(velocities[:, 1]),
        'std_velocity_x': np.std(velocities[:, 0]),
        'std_velocity_y': np.std(velocities[:, 1]),
        'mean_speed': np.mean(speed),
        'max_speed': np.max(speed),
        'final_velocity': velocities[-1]
    }
    
    return results


def plot_results(kf, save_path='kalman_table_results.png'):
    """
    Plot Kalman filter results for table data.
    """
    history = np.array(kf.state_history)
    measurements = np.array(kf.measurement_history)
    timestamps = np.array(kf.timestamp_history)
    innovations = np.array(kf.innovation_history) if kf.innovation_history else None
    
    # Convert timestamps to seconds
    time = (timestamps - timestamps[0]) / 1000.0
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Trajectory
    ax = axes[0, 0]
    ax.plot(history[:, 0], history[:, 1], 'b-', alpha=0.7, label='Trajectory')
    ax.plot(history[0, 0], history[0, 1], 'go', markersize=10, label='Start')
    ax.plot(history[-1, 0], history[-1, 1], 'ro', markersize=10, label='End')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Estimated Trajectory (Constant Velocity Model)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 2: Velocity over time
    ax = axes[0, 1]
    ax.plot(time, history[:, 2], 'b-', alpha=0.7, label='X velocity')
    ax.plot(time, history[:, 3], 'r-', alpha=0.7, label='Y velocity')
    ax.axhline(y=np.mean(history[:, 2]), color='b', linestyle='--', alpha=0.5)
    ax.axhline(y=np.mean(history[:, 3]), color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity Estimates')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Speed over time
    ax = axes[0, 2]
    speed = np.linalg.norm(history[:, 2:4], axis=1)
    ax.plot(time, speed, 'g-', alpha=0.7)
    ax.axhline(y=np.mean(speed), color='g', linestyle='--', alpha=0.5, label=f'Mean: {np.mean(speed):.3f} m/s')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Speed Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Position components over time
    ax = axes[1, 0]
    ax.plot(time, history[:, 0], 'b-', label='X position')
    ax.plot(time, history[:, 1], 'r-', label='Y position')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m)')
    ax.set_title('Position Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Acceleration measurements
    ax = axes[1, 1]
    ax.plot(time[:-1], measurements[:-1, 0], 'b.', alpha=0.3, markersize=2, label='X acc')
    ax.plot(time[:-1], measurements[:-1, 1], 'r.', alpha=0.3, markersize=2, label='Y acc')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (m/s²)')
    ax.set_title('Bias-Corrected Accelerations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Innovation (if available)
    ax = axes[1, 2]
    if innovations is not None and len(innovations) > 0:
        innovations = np.array(innovations)
        # Make sure time array matches innovation length
        innovation_time = time[:len(innovations)]
        ax.plot(innovation_time, innovations[:, 0], 'b.', alpha=0.5, markersize=2, label='X innovation')
        ax.plot(innovation_time, innovations[:, 1], 'r.', alpha=0.5, markersize=2, label='Y innovation')
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Innovation (m/s²)')
        ax.set_title('Filter Innovation (Residuals)')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Innovation data not available', ha='center', va='center')
        ax.set_title('Filter Innovation')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Kalman Filter Results - Table Data (Constant Velocity)', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved to {save_path}")


def main():
    """Main function."""
    print("=" * 60)
    print("KALMAN FILTER FOR TABLE DATA")
    print("Using measured noise statistics from sensor characterization")
    print("=" * 60)
    
    # Process table data
    kf, data = process_table_data('data/table_data.txt')
    
    # Analyze results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    results = analyze_results(kf)
    
    print(f"\nTrajectory:")
    print(f"  Start position: ({results['start_position'][0]:.4f}, {results['start_position'][1]:.4f}) m")
    print(f"  End position: ({results['end_position'][0]:.4f}, {results['end_position'][1]:.4f}) m")
    print(f"  Displacement: ({results['displacement'][0]:.4f}, {results['displacement'][1]:.4f}) m")
    print(f"  Distance traveled: {results['distance']:.4f} m")
    print(f"  Total path length: {results['total_path']:.4f} m")
    
    print(f"\nVelocity (constant velocity assumption):")
    print(f"  Mean X velocity: {results['mean_velocity_x']:.4f} ± {results['std_velocity_x']:.4f} m/s")
    print(f"  Mean Y velocity: {results['mean_velocity_y']:.4f} ± {results['std_velocity_y']:.4f} m/s")
    print(f"  Mean speed: {results['mean_speed']:.4f} m/s")
    print(f"  Max speed: {results['max_speed']:.4f} m/s")
    print(f"  Final velocity: ({results['final_velocity'][0]:.4f}, {results['final_velocity'][1]:.4f}) m/s")
    
    # Calculate motion duration
    timestamps = np.array(kf.timestamp_history)
    duration = (timestamps[-1] - timestamps[0]) / 1000.0
    print(f"\nMotion duration: {duration:.2f} seconds")
    print(f"Average speed: {results['distance']/duration:.4f} m/s")
    
    # Generate plots
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    plot_results(kf, 'kalman_table_results.png')
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()