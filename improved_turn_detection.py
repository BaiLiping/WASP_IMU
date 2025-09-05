#!/usr/bin/env python3
"""
Improved turn detection for segmented Kalman filtering.
Uses multiple methods to detect 90-degree turns in IMU data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from read_imu_data import read_imu_data
from kalman_filter_corrected import CorrectedKalmanFilter


def analyze_acceleration_patterns(data):
    """
    Comprehensive analysis of acceleration patterns to find turns.
    """
    acc_x = data['x'].values
    acc_y = data['y'].values
    timestamps = data['timestamp'].values
    
    # Remove bias
    bias_x = np.mean(acc_x)
    bias_y = np.mean(acc_y)
    acc_x_centered = acc_x - bias_x
    acc_y_centered = acc_y - bias_y
    
    print(f"Bias: X={bias_x:.4f}, Y={bias_y:.4f}")
    
    # Create figure for analysis
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    
    # Time in seconds
    time_s = (timestamps - timestamps[0]) / 1000.0
    
    # 1. Raw acceleration
    ax = axes[0, 0]
    ax.plot(time_s, acc_x_centered, 'b-', alpha=0.5, label='X')
    ax.plot(time_s, acc_y_centered, 'r-', alpha=0.5, label='Y')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (m/s²)')
    ax.set_title('Raw Centered Acceleration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Acceleration magnitude
    acc_mag = np.sqrt(acc_x_centered**2 + acc_y_centered**2)
    ax = axes[0, 1]
    ax.plot(time_s, acc_mag, 'g-', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Magnitude (m/s²)')
    ax.set_title('Acceleration Magnitude')
    ax.grid(True, alpha=0.3)
    
    # 3. Heading angle from acceleration
    # Use smoothed acceleration to get heading
    window = 100  # ~3 seconds at 30Hz
    acc_x_smooth = gaussian_filter1d(acc_x_centered, sigma=window/4)
    acc_y_smooth = gaussian_filter1d(acc_y_centered, sigma=window/4)
    
    heading = np.arctan2(acc_y_smooth, acc_x_smooth)
    heading_deg = np.degrees(heading)
    
    ax = axes[1, 0]
    ax.plot(time_s, heading_deg, 'b-', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Heading (degrees)')
    ax.set_title('Estimated Heading from Smoothed Acceleration')
    ax.grid(True, alpha=0.3)
    
    # 4. Heading change rate (angular velocity)
    heading_unwrapped = np.unwrap(heading)
    heading_change = np.gradient(heading_unwrapped) * 30  # Approximate sample rate
    heading_change_smooth = gaussian_filter1d(np.abs(heading_change), sigma=window/4)
    
    ax = axes[1, 1]
    ax.plot(time_s, heading_change_smooth, 'r-', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angular Rate (rad/s)')
    ax.set_title('Smoothed Angular Rate (Turn Indicator)')
    ax.grid(True, alpha=0.3)
    
    # 5. Jerk (rate of change of acceleration)
    jerk_x = np.gradient(acc_x_centered) * 30
    jerk_y = np.gradient(acc_y_centered) * 30
    jerk_mag = np.sqrt(jerk_x**2 + jerk_y**2)
    jerk_smooth = gaussian_filter1d(jerk_mag, sigma=window/8)
    
    ax = axes[2, 0]
    ax.plot(time_s, jerk_smooth, 'm-', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Jerk (m/s³)')
    ax.set_title('Smoothed Jerk Magnitude')
    ax.grid(True, alpha=0.3)
    
    # 6. Dominant axis changes
    # Check which axis has larger absolute acceleration
    dominant_axis = np.where(np.abs(acc_x_smooth) > np.abs(acc_y_smooth), 1, -1)
    axis_changes = np.diff(dominant_axis)
    axis_change_points = np.where(np.abs(axis_changes) > 0)[0]
    
    ax = axes[2, 1]
    ax.plot(time_s, dominant_axis, 'g-', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Dominant Axis (+1=X, -1=Y)')
    ax.set_title('Dominant Acceleration Axis')
    ax.grid(True, alpha=0.3)
    
    # 7. Running standard deviation (activity level)
    window_std = 150  # 5 seconds
    running_std_x = pd.Series(acc_x_centered).rolling(window=window_std, center=True).std()
    running_std_y = pd.Series(acc_y_centered).rolling(window=window_std, center=True).std()
    activity_level = np.sqrt(running_std_x**2 + running_std_y**2)
    
    ax = axes[3, 0]
    ax.plot(time_s, activity_level, 'c-', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Activity Level')
    ax.set_title('Running Std Dev (Activity Indicator)')
    ax.grid(True, alpha=0.3)
    
    # 8. Combined turn score
    # Normalize each indicator
    turn_score = (
        heading_change_smooth / np.max(heading_change_smooth) * 0.4 +
        jerk_smooth / np.max(jerk_smooth) * 0.3 +
        activity_level.fillna(0).values / np.nanmax(activity_level) * 0.3
    )
    
    ax = axes[3, 1]
    ax.plot(time_s, turn_score, 'k-', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Turn Score')
    ax.set_title('Combined Turn Detection Score')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Turn Detection Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig('turn_detection_analysis.png', dpi=150)
    plt.show()
    
    return {
        'time_s': time_s,
        'acc_x_centered': acc_x_centered,
        'acc_y_centered': acc_y_centered,
        'heading_change': heading_change_smooth,
        'jerk': jerk_smooth,
        'activity': activity_level.fillna(0).values,
        'turn_score': turn_score,
        'bias_x': bias_x,
        'bias_y': bias_y
    }


def find_turning_points(analysis_data, min_distance_seconds=20):
    """
    Find turning points from analyzed data.
    
    Parameters:
    -----------
    analysis_data : dict
        Output from analyze_acceleration_patterns
    min_distance_seconds : float
        Minimum time between turns
        
    Returns:
    --------
    list
        Indices of turning points
    """
    turn_score = analysis_data['turn_score']
    time_s = analysis_data['time_s']
    
    # Find peaks in turn score
    from scipy.signal import find_peaks
    
    # Set threshold based on statistics
    threshold = np.mean(turn_score) + 1.5 * np.std(turn_score)
    
    # Find peaks with minimum distance
    min_distance_samples = int(min_distance_seconds * 30)  # Assuming ~30Hz
    peaks, properties = find_peaks(turn_score, 
                                  height=threshold,
                                  distance=min_distance_samples,
                                  prominence=0.1)
    
    # If we found too many peaks, select the most prominent
    if len(peaks) > 2:
        # Sort by prominence and select top 2
        prominences = properties['prominences']
        top_2_idx = np.argsort(prominences)[-2:]
        peaks = peaks[top_2_idx]
        peaks = np.sort(peaks)
    
    print(f"\nFound {len(peaks)} turning points:")
    for i, peak in enumerate(peaks):
        print(f"  Turn {i+1}: at {time_s[peak]:.1f} seconds (sample {peak})")
    
    # Plot turn detection result
    plt.figure(figsize=(12, 6))
    plt.plot(time_s, turn_score, 'b-', alpha=0.7, label='Turn Score')
    plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label='Threshold')
    
    for i, peak in enumerate(peaks):
        plt.axvline(x=time_s[peak], color='g', linestyle='--', alpha=0.7)
        plt.text(time_s[peak], plt.ylim()[1]*0.9, f'Turn {i+1}', 
                rotation=90, ha='right', fontweight='bold')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Turn Score')
    plt.title('Detected Turning Points')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('detected_turns.png', dpi=150)
    plt.show()
    
    return peaks


def segment_and_filter(data, turning_points, analysis_data):
    """
    Segment data and apply Kalman filter to each segment.
    """
    acc_x = analysis_data['acc_x_centered']
    acc_y = analysis_data['acc_y_centered']
    timestamps = data['timestamp'].values
    
    # Create segments
    segments = []
    segment_starts = [0] + list(turning_points)
    segment_ends = list(turning_points) + [len(data)]
    
    print(f"\n{'='*60}")
    print("PROCESSING SEGMENTS")
    print('='*60)
    
    results = []
    cumulative_position = np.array([0.0, 0.0])
    cumulative_velocity = np.array([0.0, 0.0])
    
    for seg_idx, (start, end) in enumerate(zip(segment_starts, segment_ends)):
        print(f"\nSegment {seg_idx + 1}: samples {start}-{end}")
        
        # Extract segment
        seg_acc_x = acc_x[start:end]
        seg_acc_y = acc_y[start:end]
        seg_timestamps = timestamps[start:end]
        
        # Calculate dt
        dt_ms = np.diff(seg_timestamps)
        if len(dt_ms) > 0:
            dt = np.median(dt_ms) / 1000.0
        else:
            dt = 0.033  # Default
        
        duration = (seg_timestamps[-1] - seg_timestamps[0]) / 1000.0
        
        print(f"  Duration: {duration:.1f} s")
        print(f"  Samples: {len(seg_acc_x)}")
        
        # Estimate noise
        measurement_noise = np.sqrt(np.mean([np.var(seg_acc_x), np.var(seg_acc_y)]))
        process_noise = measurement_noise * 0.1
        
        # Create Kalman filter
        kf = CorrectedKalmanFilter(dt=dt,
                                  process_noise_acc=process_noise,
                                  measurement_noise=measurement_noise)
        
        # Set initial state
        kf.state[0:2] = cumulative_position
        kf.state[2:4] = cumulative_velocity
        
        # Process measurements
        for i in range(len(seg_acc_x)):
            measurement = [seg_acc_x[i], seg_acc_y[i]]
            kf.filter_step(measurement)
        
        # Get results
        history = np.array(kf.state_history)
        positions = history[:, 0:2]
        
        # Calculate metrics
        path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        displacement = np.linalg.norm(positions[-1] - positions[0])
        
        # Update cumulative state
        cumulative_position = positions[-1]
        cumulative_velocity = history[-1, 2:4]
        
        result = {
            'segment': seg_idx + 1,
            'start': start,
            'end': end,
            'duration': duration,
            'samples': len(seg_acc_x),
            'path_length': path_length,
            'displacement': displacement,
            'history': history,
            'start_pos': positions[0],
            'end_pos': positions[-1]
        }
        results.append(result)
        
        print(f"  Path length: {path_length:.2f} m")
        print(f"  Displacement: {displacement:.2f} m")
    
    return results


def visualize_segments(results):
    """Create comprehensive visualization of segmented results."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    # 1. Combined trajectory
    ax = axes[0, 0]
    for i, res in enumerate(results):
        history = res['history']
        color = colors[i % len(colors)]
        ax.plot(history[:, 0], history[:, 1], color=color, alpha=0.7,
                linewidth=2, label=f"Segment {res['segment']}")
        ax.plot(history[0, 0], history[0, 1], 'o', color=color, markersize=10)
        ax.plot(history[-1, 0], history[-1, 1], 's', color=color, markersize=10)
    
    ax.plot(0, 0, 'k*', markersize=15, label='Origin')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Complete Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 2-4. Individual segments
    for i in range(min(3, len(results))):
        ax = axes[0, i+1] if i < 2 else axes[1, i-2]
        res = results[i]
        history = res['history']
        
        # Center at segment start for clarity
        positions_centered = history[:, 0:2] - history[0, 0:2]
        
        ax.plot(positions_centered[:, 0], positions_centered[:, 1], 
                'b-', alpha=0.7, linewidth=2)
        ax.plot(0, 0, 'go', markersize=10, label='Start')
        ax.plot(positions_centered[-1, 0], positions_centered[-1, 1], 
                'ro', markersize=10, label='End')
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(f"Segment {res['segment']} ({res['duration']:.1f}s, {res['samples']} samples)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    # 5. Distance metrics bar chart
    ax = axes[1, 1]
    segment_labels = [f"Seg {r['segment']}" for r in results]
    path_lengths = [r['path_length'] for r in results]
    displacements = [r['displacement'] for r in results]
    
    x = np.arange(len(segment_labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, path_lengths, width, label='Path Length', 
                   color='skyblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, displacements, width, label='Displacement',
                   color='lightcoral', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Segment')
    ax.set_ylabel('Distance (m)')
    ax.set_title('Distance Metrics by Segment')
    ax.set_xticks(x)
    ax.set_xticklabels(segment_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    total_path = sum(r['path_length'] for r in results)
    total_displacement = np.linalg.norm(results[-1]['end_pos'] - results[0]['start_pos'])
    total_duration = sum(r['duration'] for r in results)
    
    summary_text = f"""SUMMARY STATISTICS
    
Total Path Length: {total_path:.2f} m
Total Displacement: {total_displacement:.2f} m
Total Duration: {total_duration:.1f} s
Path Efficiency: {total_displacement/total_path*100:.1f}%

Segment Details:
"""
    for res in results:
        avg_speed = res['path_length'] / res['duration']
        summary_text += f"\nSeg {res['segment']}: {res['path_length']:.1f}m in {res['duration']:.1f}s ({avg_speed:.2f}m/s)"
    
    summary_text += f"\n\nOverall Avg Speed: {total_path/total_duration:.2f} m/s"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('Segmented Trajectory Analysis - Run2', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('improved_segmentation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nVisualization saved to improved_segmentation.png")


def main():
    """Main function for improved turn detection and segmentation."""
    
    print("="*60)
    print("IMPROVED TURN DETECTION AND SEGMENTATION")
    print("="*60)
    
    # Load data
    data = read_imu_data('data/run2.txt')
    print(f"\nLoaded {len(data)} samples")
    print(f"Duration: {(data['timestamp'].max() - data['timestamp'].min())/1000:.1f} seconds")
    
    # Analyze acceleration patterns
    print("\nAnalyzing acceleration patterns...")
    analysis = analyze_acceleration_patterns(data)
    
    # Find turning points
    turning_points = find_turning_points(analysis, min_distance_seconds=30)
    
    # If we don't find exactly 2 turns, allow manual override
    if len(turning_points) != 2:
        print(f"\nWarning: Found {len(turning_points)} turns instead of 2")
        print("Would you like to specify turn points manually?")
        print("Looking at the plots, estimate the time (in seconds) of each turn.")
        
        # For now, use automatic detection or fallback
        if len(turning_points) < 2:
            # Use thirds as fallback
            n = len(data)
            turning_points = np.array([n//3, 2*n//3])
            print(f"Using fallback: dividing into thirds at samples {turning_points}")
    
    # Segment and filter
    results = segment_and_filter(data, turning_points, analysis)
    
    # Final results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print('='*60)
    
    total_path = sum(r['path_length'] for r in results)
    total_displacement = np.linalg.norm(results[-1]['end_pos'] - results[0]['start_pos'])
    
    print(f"\nTotal path length: {total_path:.2f} m")
    print(f"Total displacement: {total_displacement:.2f} m")
    print(f"Path efficiency: {total_displacement/total_path*100:.1f}%")
    
    # Visualize
    visualize_segments(results)
    
    return results, analysis


if __name__ == "__main__":
    results, analysis = main()