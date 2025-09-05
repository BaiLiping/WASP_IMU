#!/usr/bin/env python3
"""
Simplified segmented Kalman filter for run1 data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from read_imu_data import read_imu_data
from kalman_filter_corrected import CorrectedKalmanFilter


def simple_turn_detection(data):
    """
    Simplified turn detection based on acceleration patterns.
    """
    acc_x = data['x'].values
    acc_y = data['y'].values
    timestamps = data['timestamp'].values
    
    # Remove bias
    acc_x = acc_x - np.mean(acc_x)
    acc_y = acc_y - np.mean(acc_y)
    
    # Time in seconds
    time_s = (timestamps - timestamps[0]) / 1000.0
    
    # Smooth the acceleration
    window = 50
    acc_x_smooth = gaussian_filter1d(acc_x, sigma=window/4)
    acc_y_smooth = gaussian_filter1d(acc_y, sigma=window/4)
    
    # Calculate heading angle
    heading = np.arctan2(acc_y_smooth, acc_x_smooth)
    heading_unwrapped = np.unwrap(heading)
    
    # Calculate angular rate
    angular_rate = np.abs(np.gradient(heading_unwrapped))
    angular_rate_smooth = gaussian_filter1d(angular_rate, sigma=window)
    
    # Find peaks in angular rate
    threshold = np.mean(angular_rate_smooth) + 2 * np.std(angular_rate_smooth)
    peaks, _ = find_peaks(angular_rate_smooth, 
                         height=threshold,
                         distance=500)  # At least 500 samples apart
    
    # If we find more than 2 peaks, keep the 2 highest
    if len(peaks) > 2:
        peak_heights = angular_rate_smooth[peaks]
        top_2_idx = np.argsort(peak_heights)[-2:]
        peaks = peaks[top_2_idx]
        peaks = np.sort(peaks)
    
    print(f"Detected {len(peaks)} turning points at times:")
    for i, peak in enumerate(peaks):
        print(f"  Turn {i+1}: {time_s[peak]:.1f} seconds (sample {peak})")
    
    # Simple plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Acceleration
    ax = axes[0]
    ax.plot(time_s, acc_x, 'b-', alpha=0.3, label='X accel')
    ax.plot(time_s, acc_y, 'r-', alpha=0.3, label='Y accel')
    ax.plot(time_s, acc_x_smooth, 'b-', alpha=0.8, label='X smooth')
    ax.plot(time_s, acc_y_smooth, 'r-', alpha=0.8, label='Y smooth')
    for peak in peaks:
        ax.axvline(x=time_s[peak], color='g', linestyle='--', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (m/s²)')
    ax.set_title('Run1 Acceleration Data')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Angular rate
    ax = axes[1]
    ax.plot(time_s, angular_rate_smooth, 'k-', alpha=0.7)
    ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label='Threshold')
    for i, peak in enumerate(peaks):
        ax.axvline(x=time_s[peak], color='g', linestyle='--', alpha=0.7)
        ax.text(time_s[peak], ax.get_ylim()[1]*0.9, f'Turn {i+1}', 
               rotation=90, ha='right')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angular Rate')
    ax.set_title('Turn Detection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('run1_turn_detection.png', dpi=150)
    plt.close()
    
    return peaks, acc_x, acc_y


def process_segments(data, turning_points, acc_x, acc_y):
    """
    Process each segment with Kalman filter.
    """
    timestamps = data['timestamp'].values
    
    # Create segments
    segment_starts = [0] + list(turning_points)
    segment_ends = list(turning_points) + [len(data)]
    
    results = []
    cumulative_position = np.array([0.0, 0.0])
    cumulative_velocity = np.array([0.0, 0.0])
    
    print(f"\nProcessing {len(segment_starts)} segments:")
    
    for seg_idx, (start, end) in enumerate(zip(segment_starts, segment_ends)):
        # Extract segment
        seg_acc_x = acc_x[start:end]
        seg_acc_y = acc_y[start:end]
        seg_timestamps = timestamps[start:end]
        
        # Calculate dt
        if len(seg_timestamps) > 1:
            dt = np.median(np.diff(seg_timestamps)) / 1000.0
            duration = (seg_timestamps[-1] - seg_timestamps[0]) / 1000.0
        else:
            continue
        
        print(f"\nSegment {seg_idx + 1}:")
        print(f"  Samples: {start} to {end} ({end-start} samples)")
        print(f"  Duration: {duration:.1f} seconds")
        print(f"  dt: {dt*1000:.2f} ms")
        
        # Estimate noise
        measurement_noise = np.sqrt(np.mean([np.var(seg_acc_x), np.var(seg_acc_y)]))
        process_noise = measurement_noise * 0.1
        
        # Create filter
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
        
        results.append({
            'segment': seg_idx + 1,
            'duration': duration,
            'path_length': path_length,
            'displacement': displacement,
            'history': history
        })
        
        print(f"  Path length: {path_length:.2f} m")
        print(f"  Displacement: {displacement:.2f} m")
    
    return results


def visualize_results(results):
    """
    Visualize segmented results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    colors = ['blue', 'green', 'red', 'purple']
    
    # Combined trajectory
    ax = axes[0, 0]
    for i, res in enumerate(results):
        history = res['history']
        color = colors[i % len(colors)]
        ax.plot(history[:, 0], history[:, 1], color=color, alpha=0.7,
               linewidth=2, label=f"Segment {res['segment']}")
        ax.plot(history[0, 0], history[0, 1], 'o', color=color, markersize=8)
        ax.plot(history[-1, 0], history[-1, 1], 's', color=color, markersize=8)
    
    ax.plot(0, 0, 'k*', markersize=12, label='Origin')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Run1 Segmented Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Velocity magnitude over time
    ax = axes[0, 1]
    time_offset = 0
    for i, res in enumerate(results):
        history = res['history']
        speed = np.linalg.norm(history[:, 2:4], axis=1)
        time = np.arange(len(speed)) * 0.033 + time_offset  # Approximate dt
        ax.plot(time, speed, colors[i % len(colors)], alpha=0.7,
               label=f"Segment {res['segment']}")
        time_offset = time[-1]
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Speed over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Distance metrics
    ax = axes[1, 0]
    segment_labels = [f"Seg {r['segment']}" for r in results]
    path_lengths = [r['path_length'] for r in results]
    displacements = [r['displacement'] for r in results]
    
    x = np.arange(len(segment_labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, path_lengths, width, label='Path Length',
                  color='skyblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, displacements, width, label='Displacement',
                  color='lightcoral', alpha=0.8)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{height:.1f}', ha='center', va='bottom')
    
    ax.set_xlabel('Segment')
    ax.set_ylabel('Distance (m)')
    ax.set_title('Distance by Segment')
    ax.set_xticks(x)
    ax.set_xticklabels(segment_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Summary
    ax = axes[1, 1]
    ax.axis('off')
    
    total_path = sum(r['path_length'] for r in results)
    total_duration = sum(r['duration'] for r in results)
    
    if len(results) > 0:
        total_displacement = np.linalg.norm(
            results[-1]['history'][-1, 0:2] - results[0]['history'][0, 0:2]
        )
    else:
        total_displacement = 0
    
    summary = f"""RUN1 SUMMARY
    
Total Path Length: {total_path:.2f} m
Total Displacement: {total_displacement:.2f} m
Total Duration: {total_duration:.1f} s
Average Speed: {total_path/total_duration:.2f} m/s

Segments:"""
    
    for res in results:
        summary += f"\n  Seg {res['segment']}: {res['path_length']:.1f}m ({res['duration']:.1f}s)"
    
    ax.text(0.1, 0.9, summary, transform=ax.transAxes,
           fontsize=11, fontfamily='monospace', verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Run1 Segmented Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('run1_segmented_results.png', dpi=150)
    plt.close()
    
    print(f"\nResults saved to run1_segmented_results.png")


def main():
    """Main function."""
    print("="*60)
    print("SEGMENTED KALMAN FILTER FOR RUN1")
    print("="*60)
    
    # Load data
    data = read_imu_data('data/run1.txt')
    print(f"\nLoaded {len(data)} samples from run1.txt")
    print(f"Duration: {(data['timestamp'].max() - data['timestamp'].min())/1000:.1f} seconds")
    
    # Detect turns
    print("\nDetecting turns...")
    turning_points, acc_x, acc_y = simple_turn_detection(data)
    
    # If no turns detected or only 1, use equal segments
    if len(turning_points) < 2:
        print("\nUsing equal segmentation (3 segments)")
        n = len(data)
        turning_points = [n//3, 2*n//3]
    
    # Process segments
    results = process_segments(data, turning_points, acc_x, acc_y)
    
    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    total_path = sum(r['path_length'] for r in results)
    print(f"\nTotal path length: {total_path:.2f} m")
    
    if len(results) > 0:
        total_displacement = np.linalg.norm(
            results[-1]['history'][-1, 0:2] - results[0]['history'][0, 0:2]
        )
        print(f"Total displacement: {total_displacement:.2f} m")
        print(f"Path efficiency: {total_displacement/total_path*100:.1f}%")
    
    # Visualize
    visualize_results(results)
    
    return results


if __name__ == "__main__":
    results = main()