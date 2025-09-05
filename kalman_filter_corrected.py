#!/usr/bin/env python3
"""
Corrected Kalman Filter implementation for IMU acceleration data.
Carefully reviewed for mathematical correctness.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal


class CorrectedKalmanFilter:
    """
    Corrected Kalman Filter for IMU acceleration data.
    
    State vector: [x_pos, y_pos, x_vel, y_vel, x_acc, y_acc]
    Measurement vector: [x_acc, y_acc]
    
    Key corrections:
    1. Proper state transition matrix for double integration
    2. Correct process noise model
    3. Better initialization
    """
    
    def __init__(self, dt=0.01, process_noise_acc=1.0, measurement_noise=0.1):
        """
        Initialize Kalman Filter.
        
        Parameters:
        -----------
        dt : float
            Time step between measurements
        process_noise_acc : float
            Process noise for acceleration (m/s^2)^2
        measurement_noise : float
            Measurement noise variance
        """
        self.dt = dt
        
        # State vector: [x_pos, y_pos, x_vel, y_vel, x_acc, y_acc]
        self.state = np.zeros(6)
        
        # CORRECTED State transition matrix
        # Position update: p = p + v*dt + 0.5*a*dt^2
        # Velocity update: v = v + a*dt
        # Acceleration: modeled as random walk
        self.F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],         # x_pos
            [0, 1, 0, dt, 0, 0.5*dt**2],         # y_pos
            [0, 0, 1, 0, dt, 0],                 # x_vel
            [0, 0, 0, 1, 0, dt],                 # y_vel
            [0, 0, 0, 0, 1, 0],                  # x_acc (random walk)
            [0, 0, 0, 0, 0, 1]                   # y_acc (random walk)
        ])
        
        # Measurement matrix (we only measure acceleration)
        self.H = np.array([
            [0, 0, 0, 0, 1, 0],  # x_acc measurement
            [0, 0, 0, 0, 0, 1]   # y_acc measurement
        ])
        
        # CORRECTED Process noise covariance matrix
        # Using continuous white noise acceleration model
        # The process noise should account for the uncertainty in acceleration
        q = process_noise_acc  # Acceleration process noise power
        
        # Process noise matrix Q based on continuous white noise acceleration
        # This accounts for how acceleration noise propagates to position and velocity
        Q_cont = np.array([
            [dt**5/20, 0, dt**4/8, 0, dt**3/6, 0],
            [0, dt**5/20, 0, dt**4/8, 0, dt**3/6],
            [dt**4/8, 0, dt**3/3, 0, dt**2/2, 0],
            [0, dt**4/8, 0, dt**3/3, 0, dt**2/2],
            [dt**3/6, 0, dt**2/2, 0, dt, 0],
            [0, dt**3/6, 0, dt**2/2, 0, dt]
        ])
        
        # Scale by process noise power
        self.Q = q * Q_cont
        
        # For acceleration components, add additional random walk noise
        self.Q[4, 4] += process_noise_acc * dt
        self.Q[5, 5] += process_noise_acc * dt
        
        # Measurement noise covariance matrix
        self.R = np.eye(2) * measurement_noise
        
        # Initial state covariance matrix
        # Start with higher uncertainty
        self.P = np.diag([1.0, 1.0,  # position uncertainty (m^2)
                         0.1, 0.1,    # velocity uncertainty (m/s)^2
                         0.01, 0.01]) # acceleration uncertainty (m/s^2)^2
        
        # Store history for analysis
        self.state_history = []
        self.measurement_history = []
        self.innovation_history = []
        self.kalman_gain_history = []
        self.covariance_history = []
        
    def predict(self):
        """
        Prediction step of Kalman filter.
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
        # Convert measurement to numpy array
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
        
        # Update covariance using Joseph form for numerical stability
        I_KH = np.eye(6) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        # Store for diagnostics
        self.innovation_history.append(y.flatten())
        self.kalman_gain_history.append(K.copy())
        
    def filter_step(self, measurement):
        """
        Complete filter step: predict then update.
        """
        self.predict()
        self.update(measurement)
        
        # Store history
        self.state_history.append(self.state.copy())
        self.measurement_history.append(measurement)
        self.covariance_history.append(np.diag(self.P).copy())


def analyze_run2_data():
    """Detailed analysis of run2 data."""
    from read_imu_data import read_imu_data
    
    print("=" * 60)
    print("DETAILED ANALYSIS OF RUN2 DATA")
    print("=" * 60)
    
    # Read data
    data = read_imu_data('data/run2.txt')
    print(f"\nData shape: {data.shape}")
    print(f"Duration: {(data['timestamp'].max() - data['timestamp'].min())/1000:.2f} seconds")
    
    # Analyze timestamps
    timestamps = data['timestamp'].values
    dt_ms = np.diff(timestamps)
    dt_s = dt_ms / 1000.0
    
    print(f"\nTimestamp Analysis:")
    print(f"  Mean dt: {np.mean(dt_s)*1000:.2f} ms ({np.mean(dt_s):.4f} s)")
    print(f"  Median dt: {np.median(dt_s)*1000:.2f} ms ({np.median(dt_s):.4f} s)")
    print(f"  Std dt: {np.std(dt_s)*1000:.2f} ms")
    print(f"  Min dt: {np.min(dt_s)*1000:.2f} ms")
    print(f"  Max dt: {np.max(dt_s)*1000:.2f} ms")
    
    # Analyze acceleration data
    acc_x = data['x'].values
    acc_y = data['y'].values
    
    print(f"\nRaw Acceleration Statistics:")
    print(f"  X: mean={np.mean(acc_x):.4f}, std={np.std(acc_x):.4f}, min={np.min(acc_x):.4f}, max={np.max(acc_x):.4f}")
    print(f"  Y: mean={np.mean(acc_y):.4f}, std={np.std(acc_y):.4f}, min={np.min(acc_y):.4f}, max={np.max(acc_y):.4f}")
    
    # Remove bias
    bias_x = np.mean(acc_x)
    bias_y = np.mean(acc_y)
    acc_x_debiased = acc_x - bias_x
    acc_y_debiased = acc_y - bias_y
    
    print(f"\nBias Removed:")
    print(f"  X bias: {bias_x:.4f} m/s²")
    print(f"  Y bias: {bias_y:.4f} m/s²")
    
    print(f"\nDebiased Acceleration Statistics:")
    print(f"  X: std={np.std(acc_x_debiased):.4f}, min={np.min(acc_x_debiased):.4f}, max={np.max(acc_x_debiased):.4f}")
    print(f"  Y: std={np.std(acc_y_debiased):.4f}, min={np.min(acc_y_debiased):.4f}, max={np.max(acc_y_debiased):.4f}")
    
    # Estimate noise characteristics
    # Use Allan variance or simply compute autocorrelation
    print(f"\nNoise Characteristics:")
    
    # Compute autocorrelation to check for white noise
    from scipy.stats import normaltest
    
    # Test for normality (white noise should be normally distributed)
    _, p_value_x = normaltest(acc_x_debiased)
    _, p_value_y = normaltest(acc_y_debiased)
    
    print(f"  Normality test p-values: X={p_value_x:.4f}, Y={p_value_y:.4f}")
    if p_value_x < 0.05 or p_value_y < 0.05:
        print("  Warning: Data may not be normally distributed (not pure white noise)")
    
    # Check for outliers
    outliers_x = np.sum(np.abs(acc_x_debiased) > 3 * np.std(acc_x_debiased))
    outliers_y = np.sum(np.abs(acc_y_debiased) > 3 * np.std(acc_y_debiased))
    print(f"  Outliers (>3σ): X={outliers_x} ({100*outliers_x/len(acc_x_debiased):.1f}%), Y={outliers_y} ({100*outliers_y/len(acc_y_debiased):.1f}%)")
    
    return data, acc_x_debiased, acc_y_debiased, np.median(dt_s)


def process_run2_corrected():
    """Process run2 with corrected Kalman filter."""
    
    # Analyze data first
    data, acc_x_debiased, acc_y_debiased, dt_median = analyze_run2_data()
    
    print("\n" + "=" * 60)
    print("PROCESSING RUN2 WITH CORRECTED KALMAN FILTER")
    print("=" * 60)
    
    # Estimate measurement noise from data
    measurement_noise = np.sqrt(np.mean([np.var(acc_x_debiased), np.var(acc_y_debiased)]))
    print(f"\nEstimated measurement noise: {measurement_noise:.4f} (m/s²)²")
    
    # Process noise should be smaller than measurement noise for smooth estimates
    # But not too small or the filter won't track changes
    process_noise_acc = measurement_noise * 0.1  # Tune this parameter
    
    print(f"Using process noise: {process_noise_acc:.4f} (m/s²)²")
    print(f"Using dt: {dt_median:.4f} seconds")
    
    # Initialize filter
    kf = CorrectedKalmanFilter(dt=dt_median, 
                               process_noise_acc=process_noise_acc,
                               measurement_noise=measurement_noise)
    
    # Process measurements
    print(f"\nProcessing {len(data)} measurements...")
    for i in range(len(data)):
        measurement = [acc_x_debiased[i], acc_y_debiased[i]]
        kf.filter_step(measurement)
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(data)} samples")
    
    print(f"Processing complete!")
    
    # Calculate results
    history = np.array(kf.state_history)
    
    # Extract final estimates
    final_pos = history[-1, 0:2]
    final_vel = history[-1, 2:4]
    
    print(f"\nFinal State Estimates:")
    print(f"  Position: ({final_pos[0]:.4f}, {final_pos[1]:.4f}) m")
    print(f"  Velocity: ({final_vel[0]:.4f}, {final_vel[1]:.4f}) m/s")
    
    # Calculate distances
    positions = history[:, 0:2]
    euclidean_dist = np.linalg.norm(final_pos)
    path_distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    total_path = np.sum(path_distances)
    max_displacement = np.max(np.linalg.norm(positions, axis=1))
    
    print(f"\nDistance Metrics:")
    print(f"  Euclidean distance from origin: {euclidean_dist:.4f} m")
    print(f"  Total path length: {total_path:.4f} m")
    print(f"  Maximum displacement: {max_displacement:.4f} m")
    
    # Check filter health
    final_covariance = np.array(kf.covariance_history[-1])
    print(f"\nFinal Uncertainty (1σ):")
    print(f"  Position: X=±{np.sqrt(final_covariance[0]):.4f} m, Y=±{np.sqrt(final_covariance[1]):.4f} m")
    print(f"  Velocity: X=±{np.sqrt(final_covariance[2]):.4f} m/s, Y=±{np.sqrt(final_covariance[3]):.4f} m/s")
    print(f"  Acceleration: X=±{np.sqrt(final_covariance[4]):.4f} m/s², Y=±{np.sqrt(final_covariance[5]):.4f} m/s²")
    
    return kf, data


def create_diagnostic_plots(kf, data):
    """Create detailed diagnostic plots."""
    
    history = np.array(kf.state_history)
    measurements = np.array(kf.measurement_history)
    innovations = np.array(kf.innovation_history)
    covariances = np.array(kf.covariance_history)
    
    # Time array
    time = np.arange(len(history)) * kf.dt
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # 1. Trajectory
    ax = axes[0, 0]
    ax.plot(history[:, 0], history[:, 1], 'b-', alpha=0.6)
    ax.plot(0, 0, 'go', markersize=10, label='Start')
    ax.plot(history[-1, 0], history[-1, 1], 'ro', markersize=10, label='End')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Estimated Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 2. Position over time with uncertainty
    ax = axes[0, 1]
    ax.plot(time, history[:, 0], 'b-', alpha=0.8, label='X')
    ax.fill_between(time, 
                     history[:, 0] - np.sqrt(covariances[:, 0]),
                     history[:, 0] + np.sqrt(covariances[:, 0]),
                     alpha=0.2, color='blue')
    ax.plot(time, history[:, 1], 'r-', alpha=0.8, label='Y')
    ax.fill_between(time,
                     history[:, 1] - np.sqrt(covariances[:, 1]),
                     history[:, 1] + np.sqrt(covariances[:, 1]),
                     alpha=0.2, color='red')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m)')
    ax.set_title('Position with Uncertainty (1σ)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Velocity
    ax = axes[0, 2]
    ax.plot(time, history[:, 2], 'b-', alpha=0.8, label='X velocity')
    ax.plot(time, history[:, 3], 'r-', alpha=0.8, label='Y velocity')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity Estimates')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Acceleration (measured vs filtered)
    ax = axes[1, 0]
    ax.plot(time, measurements[:, 0], 'b.', alpha=0.1, markersize=1, label='Measured X')
    ax.plot(time, history[:, 4], 'b-', alpha=0.8, label='Filtered X')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('X Acceleration (m/s²)')
    ax.set_title('X Acceleration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Y Acceleration
    ax = axes[1, 1]
    ax.plot(time, measurements[:, 1], 'r.', alpha=0.1, markersize=1, label='Measured Y')
    ax.plot(time, history[:, 5], 'r-', alpha=0.8, label='Filtered Y')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Y Acceleration (m/s²)')
    ax.set_title('Y Acceleration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Innovation (measurement residual)
    ax = axes[1, 2]
    ax.plot(time, innovations[:, 0], 'b-', alpha=0.5, label='X innovation')
    ax.plot(time, innovations[:, 1], 'r-', alpha=0.5, label='Y innovation')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Innovation (m/s²)')
    ax.set_title('Measurement Innovation (Should be Zero-Mean)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 7. Covariance evolution
    ax = axes[2, 0]
    ax.semilogy(time, covariances[:, 0], 'b-', label='X pos')
    ax.semilogy(time, covariances[:, 1], 'r-', label='Y pos')
    ax.semilogy(time, covariances[:, 2], 'b--', label='X vel')
    ax.semilogy(time, covariances[:, 3], 'r--', label='Y vel')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Variance')
    ax.set_title('Covariance Evolution (Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 8. Speed magnitude
    ax = axes[2, 1]
    speed = np.linalg.norm(history[:, 2:4], axis=1)
    ax.plot(time, speed, 'm-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Speed Magnitude')
    ax.grid(True, alpha=0.3)
    
    # 9. Distance from origin
    ax = axes[2, 2]
    distance = np.linalg.norm(history[:, 0:2], axis=1)
    ax.plot(time, distance, 'g-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance (m)')
    ax.set_title('Distance from Origin')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Run2 Kalman Filter Diagnostics', fontsize=16)
    plt.tight_layout()
    plt.savefig('run2_diagnostics.png', dpi=150)
    plt.close()
    
    print("\nDiagnostic plots saved to run2_diagnostics.png")
    
    # Check innovation statistics
    print("\nInnovation Statistics (should be zero-mean white noise):")
    print(f"  X: mean={np.mean(innovations[:, 0]):.6f}, std={np.std(innovations[:, 0]):.4f}")
    print(f"  Y: mean={np.mean(innovations[:, 1]):.6f}, std={np.std(innovations[:, 1]):.4f}")


if __name__ == "__main__":
    kf, data = process_run2_corrected()
    
    print("\n" + "=" * 60)
    print("CREATING DIAGNOSTIC PLOTS")
    print("=" * 60)
    create_diagnostic_plots(kf, data)