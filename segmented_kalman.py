#!/usr/bin/env python3
"""
Segmented Kalman Filter for run2 data with turn detection.
Segments the data at 90-degree turns to reduce drift accumulation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import uniform_filter1d
from read_imu_data import read_imu_data
from kalman_filter_corrected import CorrectedKalmanFilter


def detect_turns(acc_x, acc_y, window_size=150, threshold_factor=2.0):
    """
    Detect turning points in acceleration data.
    
    Turns are characterized by:
    1. Changes in acceleration direction
    2. Increased acceleration magnitude during turning
    3. Changes in the dominant acceleration axis
    
    Parameters:
    -----------
    acc_x, acc_y : np.array
        Debiased acceleration data
    window_size : int
        Window size for moving statistics (samples)
    threshold_factor : float
        Factor for determining significant changes
    
    Returns:
    --------
    list
        Indices of detected turning points
    """
    # Compute acceleration magnitude
    acc_mag = np.sqrt(acc_x**2 + acc_y**2)
    
    # Compute moving average and std
    acc_mag_smooth = uniform_filter1d(acc_mag, size=window_size, mode='nearest')
    
    # Compute local variance (indicator of turning)
    local_var = uniform_filter1d((acc_mag - acc_mag_smooth)**2, size=window_size, mode='nearest')
    
    # Compute direction changes using acceleration angle
    acc_angle = np.arctan2(acc_y, acc_x)
    
    # Unwrap angle to avoid discontinuities
    acc_angle_unwrapped = np.unwrap(acc_angle)
    
    # Compute angular velocity (rate of direction change)
    angular_change = np.abs(np.diff(acc_angle_unwrapped))
    angular_change = np.insert(angular_change, 0, 0)  # Pad to maintain array size
    
    # Smooth angular change
    angular_change_smooth = uniform_filter1d(angular_change, size=window_size, mode='nearest')
    
    # Find peaks in both variance and angular change
    # These indicate turning events
    var_threshold = np.mean(local_var) + threshold_factor * np.std(local_var)
    angular_threshold = np.mean(angular_change_smooth) + threshold_factor * np.std(angular_change_smooth)
    
    # Find peaks
    from scipy.signal import find_peaks
    
    # Find peaks in variance
    var_peaks, _ = find_peaks(local_var, height=var_threshold, distance=500)
    
    # Find peaks in angular change
    angular_peaks, _ = find_peaks(angular_change_smooth, height=angular_threshold, distance=500)
    
    # Combine and sort peaks
    all_peaks = np.unique(np.concatenate([var_peaks, angular_peaks]))
    
    # If we have too many peaks, select the most prominent ones
    if len(all_peaks) > 2:
        # Score each peak by combined metric
        scores = local_var[all_peaks] / np.mean(local_var) + angular_change_smooth[all_peaks] / np.mean(angular_change_smooth)
        # Select top 2
        top_indices = np.argsort(scores)[-2:]
        all_peaks = all_peaks[top_indices]
        all_peaks = np.sort(all_peaks)
    
    print(f"Detected {len(all_peaks)} turning points at samples: {all_peaks}")
    
    return all_peaks, acc_mag, local_var, angular_change_smooth


def segment_and_process(data, turning_points):
    """
    Segment data at turning points and process each segment with Kalman filter.
    
    Parameters:
    -----------
    data : pd.DataFrame
        IMU data
    turning_points : list
        Indices of turning points
    
    Returns:
    --------
    dict
        Results for each segment
    """
    # Get acceleration data
    acc_x_raw = data['x'].values
    acc_y_raw = data['y'].values
    timestamps = data['timestamp'].values
    
    # Remove bias
    bias_x = np.mean(acc_x_raw)
    bias_y = np.mean(acc_y_raw)
    acc_x = acc_x_raw - bias_x
    acc_y = acc_y_raw - bias_y
    
    print(f"\nBias removed: X={bias_x:.4f}, Y={bias_y:.4f}")
    
    # Create segments
    segments = []
    segment_starts = [0] + list(turning_points)
    segment_ends = list(turning_points) + [len(data)]
    
    for i, (start, end) in enumerate(zip(segment_starts, segment_ends)):
        segments.append({
            'index': i + 1,
            'start': start,
            'end': end,
            'length': end - start,
            'duration': (timestamps[end-1] - timestamps[start]) / 1000.0  # seconds
        })
    
    print(f"\nCreated {len(segments)} segments:")
    for seg in segments:
        print(f"  Segment {seg['index']}: samples {seg['start']}-{seg['end']} "
              f"({seg['length']} samples, {seg['duration']:.1f} seconds)")
    
    # Process each segment
    results = []
    cumulative_position = np.array([0.0, 0.0])
    cumulative_velocity = np.array([0.0, 0.0])
    
    for seg in segments:
        print(f"\n" + "=" * 50)
        print(f"Processing Segment {seg['index']}")
        print("=" * 50)
        
        # Extract segment data
        seg_acc_x = acc_x[seg['start']:seg['end']]
        seg_acc_y = acc_y[seg['start']:seg['end']]
        seg_timestamps = timestamps[seg['start']:seg['end']]
        
        # Calculate dt for this segment
        dt_ms = np.diff(seg_timestamps)
        dt_median = np.median(dt_ms) / 1000.0  # Convert to seconds
        
        # Estimate measurement noise for this segment
        measurement_noise = np.sqrt(np.mean([np.var(seg_acc_x), np.var(seg_acc_y)]))
        process_noise = measurement_noise * 0.1
        
        print(f"  Median dt: {dt_median*1000:.2f} ms")
        print(f"  Measurement noise: {measurement_noise:.4f}")
        
        # Initialize Kalman filter for this segment
        kf = CorrectedKalmanFilter(dt=dt_median,
                                  process_noise_acc=process_noise,
                                  measurement_noise=measurement_noise)
        
        # Set initial state from previous segment's final state
        kf.state[0:2] = cumulative_position  # Initial position
        kf.state[2:4] = cumulative_velocity  # Initial velocity
        
        # Process measurements
        for i in range(len(seg_acc_x)):
            measurement = [seg_acc_x[i], seg_acc_y[i]]
            kf.filter_step(measurement)
        
        # Get results for this segment
        history = np.array(kf.state_history)
        
        # Calculate segment metrics
        positions = history[:, 0:2]
        start_pos = positions[0]
        end_pos = positions[-1]
        
        # Path length for this segment
        path_distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        segment_path_length = np.sum(path_distances)
        
        # Displacement for this segment
        segment_displacement = np.linalg.norm(end_pos - start_pos)
        
        # Update cumulative position and velocity for next segment
        cumulative_position = end_pos
        cumulative_velocity = history[-1, 2:4]
        
        # Store results
        result = {
            'segment': seg['index'],
            'duration': seg['duration'],
            'samples': seg['length'],
            'start_pos': start_pos,
            'end_pos': end_pos,
            'displacement': segment_displacement,
            'path_length': segment_path_length,
            'history': history,
            'measurements': np.column_stack([seg_acc_x, seg_acc_y])
        }
        results.append(result)
        
        print(f"  Start position: ({start_pos[0]:.2f}, {start_pos[1]:.2f}) m")
        print(f"  End position: ({end_pos[0]:.2f}, {end_pos[1]:.2f}) m")
        print(f"  Segment displacement: {segment_displacement:.2f} m")
        print(f"  Segment path length: {segment_path_length:.2f} m")
    
    return results


def visualize_segmented_results(results, turning_points, acc_mag, local_var, angular_change):
    """Create visualization of segmented analysis."""
    
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Turn detection signals
    ax1 = fig.add_subplot(gs[0, :])
    time_samples = np.arange(len(acc_mag))
    
    ax1_twin = ax1.twinx()
    ax1.plot(time_samples, acc_mag, 'b-', alpha=0.5, label='Acc Magnitude')
    ax1.plot(time_samples, local_var, 'g-', alpha=0.7, label='Local Variance')
    ax1_twin.plot(time_samples, angular_change, 'r-', alpha=0.7, label='Angular Change')
    
    # Mark turning points
    for tp in turning_points:
        ax1.axvline(x=tp, color='k', linestyle='--', alpha=0.5)
        ax1.text(tp, ax1.get_ylim()[1]*0.9, f'Turn {turning_points.tolist().index(tp)+1}', 
                rotation=90, ha='right')
    
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Acceleration Magnitude / Variance', color='b')
    ax1_twin.set_ylabel('Angular Change Rate', color='r')
    ax1.set_title('Turn Detection Signals')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Combined trajectory
    ax2 = fig.add_subplot(gs[1, 0])
    colors = ['blue', 'green', 'red']
    
    for i, res in enumerate(results):
        history = res['history']
        color = colors[i % len(colors)]
        ax2.plot(history[:, 0], history[:, 1], color=color, alpha=0.7, 
                label=f"Segment {res['segment']}")
        ax2.plot(history[0, 0], history[0, 1], 'o', color=color, markersize=8)
        ax2.plot(history[-1, 0], history[-1, 1], 's', color=color, markersize=8)
    
    ax2.plot(0, 0, 'ko', markersize=10, label='Origin')
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_title('Segmented Trajectory')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # 3. Individual segment trajectories
    for i, res in enumerate(results):
        ax = fig.add_subplot(gs[1, i+1] if i < 2 else gs[2, i-2])
        history = res['history']
        
        # Shift to start from origin for clarity
        positions_shifted = history[:, 0:2] - history[0, 0:2]
        
        ax.plot(positions_shifted[:, 0], positions_shifted[:, 1], 'b-', alpha=0.7)
        ax.plot(0, 0, 'go', markersize=8, label='Start')
        ax.plot(positions_shifted[-1, 0], positions_shifted[-1, 1], 'ro', 
               markersize=8, label='End')
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(f"Segment {res['segment']} ({res['duration']:.1f}s)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    # 4. Metrics summary bar chart
    ax5 = fig.add_subplot(gs[2, 1])
    segment_labels = [f"Seg {r['segment']}" for r in results]
    path_lengths = [r['path_length'] for r in results]
    displacements = [r['displacement'] for r in results]
    
    x = np.arange(len(segment_labels))
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, path_lengths, width, label='Path Length', alpha=0.7)
    bars2 = ax5.bar(x + width/2, displacements, width, label='Displacement', alpha=0.7)
    
    ax5.set_xlabel('Segment')
    ax5.set_ylabel('Distance (m)')
    ax5.set_title('Segment Distances')
    ax5.set_xticks(x)
    ax5.set_xticklabels(segment_labels)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2, height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 5. Summary text
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    total_path = sum(r['path_length'] for r in results)
    total_displacement = np.linalg.norm(results[-1]['end_pos'] - results[0]['start_pos'])
    total_duration = sum(r['duration'] for r in results)
    
    summary_text = f"""SUMMARY
    
Total Path Length: {total_path:.2f} m

Total Displacement: {total_displacement:.2f} m

Total Duration: {total_duration:.1f} s

Segment Breakdown:
"""
    for res in results:
        summary_text += f"\n  Segment {res['segment']}: {res['path_length']:.1f} m ({res['duration']:.1f}s)"
    
    summary_text += f"\n\nAverage Speed: {total_path/total_duration:.2f} m/s"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Segmented Kalman Filter Analysis - Run2', fontsize=14, fontweight='bold')
    plt.savefig('segmented_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nVisualization saved to segmented_analysis.png")


def main():
    """Main function to run segmented analysis."""
    
    print("=" * 60)
    print("SEGMENTED KALMAN FILTER ANALYSIS FOR RUN2")
    print("=" * 60)
    
    # Read data
    data = read_imu_data('data/run2.txt')
    print(f"\nLoaded {len(data)} samples")
    print(f"Duration: {(data['timestamp'].max() - data['timestamp'].min())/1000:.1f} seconds")
    
    # Remove bias for turn detection
    acc_x = data['x'].values - np.mean(data['x'].values)
    acc_y = data['y'].values - np.mean(data['y'].values)
    
    # Detect turns
    print("\n" + "=" * 60)
    print("DETECTING TURNS")
    print("=" * 60)
    
    turning_points, acc_mag, local_var, angular_change = detect_turns(acc_x, acc_y)
    
    if len(turning_points) != 2:
        print(f"Warning: Expected 2 turning points, found {len(turning_points)}")
        if len(turning_points) < 2:
            # If we found fewer than 2, try to split data equally
            n_samples = len(data)
            turning_points = np.array([n_samples // 3, 2 * n_samples // 3])
            print(f"Using equally spaced points instead: {turning_points}")
    
    # Segment and process
    results = segment_and_process(data, turning_points)
    
    # Calculate total metrics
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    total_path_length = sum(r['path_length'] for r in results)
    total_displacement = np.linalg.norm(results[-1]['end_pos'] - results[0]['start_pos'])
    
    print(f"\nTotal path length (sum of segments): {total_path_length:.2f} m")
    print(f"Total displacement (start to end): {total_displacement:.2f} m")
    print(f"Path efficiency: {total_displacement/total_path_length*100:.1f}%")
    
    # Individual segment summary
    print("\nSegment Summary:")
    for res in results:
        print(f"  Segment {res['segment']}: {res['path_length']:.2f} m in {res['duration']:.1f}s "
              f"(avg speed: {res['path_length']/res['duration']:.2f} m/s)")
    
    # Create visualization
    visualize_segmented_results(results, turning_points, acc_mag, local_var, angular_change)
    
    return results


if __name__ == "__main__":
    results = main()