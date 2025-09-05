#!/usr/bin/env python3
"""
Base Kalman Filter implementation and common utilities for IMU data processing.
"""

import numpy as np
import matplotlib.pyplot as plt


def calculate_distance_traveled(state_history):
    """
    Calculate distance from start to end position.
    
    Parameters:
    -----------
    state_history : np.array or list
        State history from Kalman filter
        
    Returns:
    --------
    dict
        Dictionary with distance metrics
    """
    history = np.array(state_history)
    
    # Extract positions
    positions = history[:, 0:2]  # x and y positions
    
    # Start and end positions
    start_pos = positions[0]
    end_pos = positions[-1]
    
    # Euclidean distance
    euclidean_distance = np.linalg.norm(end_pos - start_pos)
    
    # Total path length (sum of all small movements)
    path_distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    total_path_length = np.sum(path_distances)
    
    # Maximum displacement from origin
    distances_from_origin = np.linalg.norm(positions, axis=1)
    max_displacement = np.max(distances_from_origin)
    
    results = {
        'start_position': start_pos,
        'end_position': end_pos,
        'euclidean_distance': euclidean_distance,
        'total_path_length': total_path_length,
        'max_displacement': max_displacement,
        'displacement_vector': end_pos - start_pos
    }
    
    return results


def plot_trajectory(state_history, ax, title="Position Trajectory"):
    """
    Plot the position trajectory.
    
    Parameters:
    -----------
    state_history : np.array
        State history from Kalman filter
    ax : matplotlib.axes
        Axes to plot on
    title : str
        Title for the plot
    """
    history = np.array(state_history)
    ax.plot(history[:, 0], history[:, 1], 'b-', alpha=0.7, label='Filtered trajectory')
    ax.plot(history[0, 0], history[0, 1], 'go', markersize=10, label='Start')
    ax.plot(history[-1, 0], history[-1, 1], 'ro', markersize=10, label='End')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')


def plot_velocity(state_history, time, ax):
    """
    Plot velocity over time.
    
    Parameters:
    -----------
    state_history : np.array
        State history from Kalman filter
    time : np.array
        Time array
    ax : matplotlib.axes
        Axes to plot on
    """
    history = np.array(state_history)
    ax.plot(time, history[:, 2], 'b-', alpha=0.7, label='X velocity')
    ax.plot(time, history[:, 3], 'r-', alpha=0.7, label='Y velocity')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity Estimates')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_acceleration(state_history, measurement_history, time, ax):
    """
    Plot acceleration - measured vs filtered.
    
    Parameters:
    -----------
    state_history : np.array
        State history from Kalman filter
    measurement_history : np.array
        Measurement history
    time : np.array
        Time array
    ax : matplotlib.axes
        Axes to plot on
    """
    history = np.array(state_history)
    measurements = np.array(measurement_history)
    
    ax.plot(time, measurements[:, 0], 'b.', alpha=0.3, markersize=1, label='Measured X')
    ax.plot(time, history[:, 4], 'b-', alpha=0.8, label='Filtered X')
    ax.plot(time, measurements[:, 1], 'r.', alpha=0.3, markersize=1, label='Measured Y')
    ax.plot(time, history[:, 5], 'r-', alpha=0.8, label='Filtered Y')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (m/s²)')
    ax.set_title('Acceleration: Measured vs Filtered')
    ax.legend()
    ax.grid(True, alpha=0.3)


def remove_gravity_bias(acc_x, acc_y):
    """
    Remove gravity/bias from acceleration data.
    
    Parameters:
    -----------
    acc_x : np.array
        X acceleration data
    acc_y : np.array
        Y acceleration data
        
    Returns:
    --------
    tuple
        (debiased_x, debiased_y, bias_x, bias_y)
    """
    bias_x = np.mean(acc_x)
    bias_y = np.mean(acc_y)
    return acc_x - bias_x, acc_y - bias_y, bias_x, bias_y


def estimate_measurement_noise(acc_x, acc_y):
    """
    Estimate measurement noise from acceleration data.
    
    Parameters:
    -----------
    acc_x : np.array
        X acceleration data (already debiased)
    acc_y : np.array
        Y acceleration data (already debiased)
        
    Returns:
    --------
    float
        Estimated measurement noise
    """
    acc_data = np.column_stack([acc_x, acc_y])
    measurement_cov = np.cov(acc_data.T)
    return np.sqrt(np.mean(np.diag(measurement_cov)))