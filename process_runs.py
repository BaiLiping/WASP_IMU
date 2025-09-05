#!/usr/bin/env python3
"""
Process run1 and run2 data with Kalman filter and calculate average results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from read_imu_data import read_imu_data
from kalman_filter_imu import KalmanFilter, process_imu_data_with_kalman
from kalman_base import calculate_distance_traveled, plot_trajectory


def process_both_runs():
    """Process run1 and run2 data and calculate averages."""
    
    print("=" * 60)
    print("PROCESSING RUN1 AND RUN2 WITH KALMAN FILTER")
    print("=" * 60)
    
    # Process run1
    print("\n" + "=" * 60)
    print("PROCESSING RUN1")
    print("=" * 60)
    run1_data = read_imu_data('data/run1.txt')
    print(f"Loaded {len(run1_data)} samples from run1.txt")
    print(f"Duration: {(run1_data['timestamp'].max() - run1_data['timestamp'].min())/1000:.2f} seconds")
    
    kf_run1 = process_imu_data_with_kalman(run1_data, remove_gravity=True)
    distances_run1 = calculate_distance_traveled(kf_run1.state_history)
    
    print(f"\nRun1 Results:")
    print(f"  Start position: ({distances_run1['start_position'][0]:.4f}, {distances_run1['start_position'][1]:.4f}) m")
    print(f"  End position: ({distances_run1['end_position'][0]:.4f}, {distances_run1['end_position'][1]:.4f}) m")
    print(f"  Euclidean distance: {distances_run1['euclidean_distance']:.4f} m")
    print(f"  Total path length: {distances_run1['total_path_length']:.4f} m")
    print(f"  Max displacement: {distances_run1['max_displacement']:.4f} m")
    
    # Process run2
    print("\n" + "=" * 60)
    print("PROCESSING RUN2")
    print("=" * 60)
    run2_data = read_imu_data('data/run2.txt')
    print(f"Loaded {len(run2_data)} samples from run2.txt")
    print(f"Duration: {(run2_data['timestamp'].max() - run2_data['timestamp'].min())/1000:.2f} seconds")
    
    kf_run2 = process_imu_data_with_kalman(run2_data, remove_gravity=True)
    distances_run2 = calculate_distance_traveled(kf_run2.state_history)
    
    print(f"\nRun2 Results:")
    print(f"  Start position: ({distances_run2['start_position'][0]:.4f}, {distances_run2['start_position'][1]:.4f}) m")
    print(f"  End position: ({distances_run2['end_position'][0]:.4f}, {distances_run2['end_position'][1]:.4f}) m")
    print(f"  Euclidean distance: {distances_run2['euclidean_distance']:.4f} m")
    print(f"  Total path length: {distances_run2['total_path_length']:.4f} m")
    print(f"  Max displacement: {distances_run2['max_displacement']:.4f} m")
    
    # Calculate averages
    print("\n" + "=" * 60)
    print("AVERAGE RESULTS")
    print("=" * 60)
    
    avg_euclidean = (distances_run1['euclidean_distance'] + distances_run2['euclidean_distance']) / 2
    avg_path_length = (distances_run1['total_path_length'] + distances_run2['total_path_length']) / 2
    avg_max_displacement = (distances_run1['max_displacement'] + distances_run2['max_displacement']) / 2
    
    # Calculate standard deviation for uncertainty estimate
    std_euclidean = np.std([distances_run1['euclidean_distance'], distances_run2['euclidean_distance']])
    std_path_length = np.std([distances_run1['total_path_length'], distances_run2['total_path_length']])
    std_max_displacement = np.std([distances_run1['max_displacement'], distances_run2['max_displacement']])
    
    print(f"\nAverage Euclidean distance: {avg_euclidean:.4f} ± {std_euclidean:.4f} m")
    print(f"Average total path length: {avg_path_length:.4f} ± {std_path_length:.4f} m")
    print(f"Average max displacement: {avg_max_displacement:.4f} ± {std_max_displacement:.4f} m")
    
    print(f"\nDifference between runs:")
    print(f"  Euclidean distance diff: {abs(distances_run1['euclidean_distance'] - distances_run2['euclidean_distance']):.4f} m")
    print(f"  Path length diff: {abs(distances_run1['total_path_length'] - distances_run2['total_path_length']):.4f} m")
    print(f"  Max displacement diff: {abs(distances_run1['max_displacement'] - distances_run2['max_displacement']):.4f} m")
    
    # Create comparison plot
    create_comparison_plot(kf_run1, kf_run2, distances_run1, distances_run2)
    
    return {
        'run1': distances_run1,
        'run2': distances_run2,
        'average': {
            'euclidean_distance': avg_euclidean,
            'total_path_length': avg_path_length,
            'max_displacement': avg_max_displacement,
            'std_euclidean': std_euclidean,
            'std_path_length': std_path_length,
            'std_max_displacement': std_max_displacement
        }
    }


def create_comparison_plot(kf_run1, kf_run2, distances_run1, distances_run2):
    """Create comparison visualization for run1 and run2."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Extract histories
    history1 = np.array(kf_run1.state_history)
    history2 = np.array(kf_run2.state_history)
    measurements1 = np.array(kf_run1.measurement_history)
    measurements2 = np.array(kf_run2.measurement_history)
    
    # Plot 1: Both trajectories
    ax = axes[0, 0]
    ax.plot(history1[:, 0], history1[:, 1], 'b-', alpha=0.6, label='Run1 trajectory')
    ax.plot(history1[0, 0], history1[0, 1], 'bo', markersize=8)
    ax.plot(history1[-1, 0], history1[-1, 1], 'bs', markersize=8)
    
    ax.plot(history2[:, 0], history2[:, 1], 'r-', alpha=0.6, label='Run2 trajectory')
    ax.plot(history2[0, 0], history2[0, 1], 'ro', markersize=8)
    ax.plot(history2[-1, 0], history2[-1, 1], 'rs', markersize=8)
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Position Trajectories Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 2: X acceleration comparison
    ax = axes[0, 1]
    time1 = np.arange(len(measurements1)) * kf_run1.dt
    time2 = np.arange(len(measurements2)) * kf_run2.dt
    
    ax.plot(time1, measurements1[:, 0], 'b.', alpha=0.2, markersize=1, label='Run1 measured')
    ax.plot(time1, history1[:, 4], 'b-', alpha=0.8, label='Run1 filtered')
    ax.plot(time2, measurements2[:, 0], 'r.', alpha=0.2, markersize=1, label='Run2 measured')
    ax.plot(time2, history2[:, 4], 'r-', alpha=0.8, label='Run2 filtered')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('X Acceleration (m/s²)')
    ax.set_title('X Acceleration Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Y acceleration comparison
    ax = axes[0, 2]
    ax.plot(time1, measurements1[:, 1], 'b.', alpha=0.2, markersize=1, label='Run1 measured')
    ax.plot(time1, history1[:, 5], 'b-', alpha=0.8, label='Run1 filtered')
    ax.plot(time2, measurements2[:, 1], 'r.', alpha=0.2, markersize=1, label='Run2 measured')
    ax.plot(time2, history2[:, 5], 'r-', alpha=0.8, label='Run2 filtered')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Y Acceleration (m/s²)')
    ax.set_title('Y Acceleration Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Velocity magnitude comparison
    ax = axes[1, 0]
    speed1 = np.linalg.norm(history1[:, 2:4], axis=1)
    speed2 = np.linalg.norm(history2[:, 2:4], axis=1)
    ax.plot(time1, speed1, 'b-', label='Run1')
    ax.plot(time2, speed2, 'r-', label='Run2')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Speed Magnitude Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Distance from origin
    ax = axes[1, 1]
    dist1 = np.linalg.norm(history1[:, 0:2], axis=1)
    dist2 = np.linalg.norm(history2[:, 0:2], axis=1)
    ax.plot(time1, dist1, 'b-', label='Run1')
    ax.plot(time2, dist2, 'r-', label='Run2')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance from Origin (m)')
    ax.set_title('Distance from Start Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Distance metrics comparison (bar chart)
    ax = axes[1, 2]
    metrics = ['Euclidean\nDistance', 'Total Path\nLength', 'Max\nDisplacement']
    run1_values = [distances_run1['euclidean_distance'], 
                   distances_run1['total_path_length'],
                   distances_run1['max_displacement']]
    run2_values = [distances_run2['euclidean_distance'],
                   distances_run2['total_path_length'],
                   distances_run2['max_displacement']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, run1_values, width, label='Run1', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, run2_values, width, label='Run2', color='red', alpha=0.7)
    
    ax.set_ylabel('Distance (m)')
    ax.set_title('Distance Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=8)
    
    plt.suptitle('Run1 vs Run2 Kalman Filter Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig('run_comparison.png', dpi=150)
    plt.close()
    
    print("\nPlot saved to run_comparison.png")


if __name__ == "__main__":
    results = process_both_runs()