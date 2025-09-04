#!/usr/bin/env python3
"""
Comprehensive noise analysis for IMU stationary data.
Extracts noise statistics including Allan variance, PSD, and other metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')


def allan_variance(data, dt, max_tau_ratio=0.5):
    """
    Calculate Allan variance for IMU data.
    
    Parameters:
    -----------
    data : np.array
        Time series data
    dt : float
        Sampling period in seconds
    max_tau_ratio : float
        Maximum tau as ratio of total time
        
    Returns:
    --------
    tau : np.array
        Averaging times
    avar : np.array
        Allan variance values
    adev : np.array
        Allan deviation (sqrt of variance)
    """
    n = len(data)
    max_m = int(n * max_tau_ratio)
    
    # Calculate tau values (logarithmic spacing)
    m_values = np.logspace(0, np.log10(max_m), 50).astype(int)
    m_values = np.unique(m_values)
    
    tau = m_values * dt
    avar = np.zeros(len(m_values))
    
    for i, m in enumerate(m_values):
        # Calculate averages over clusters of size m
        clusters = int(n / m)
        if clusters < 2:
            avar[i] = np.nan
            continue
            
        # Compute cluster averages
        y = np.zeros(clusters)
        for j in range(clusters):
            y[j] = np.mean(data[j*m:(j+1)*m])
        
        # Allan variance formula
        diff = np.diff(y)
        avar[i] = 0.5 * np.mean(diff**2)
    
    # Remove NaN values
    valid = ~np.isnan(avar)
    tau = tau[valid]
    avar = avar[valid]
    adev = np.sqrt(avar)
    
    return tau, avar, adev


def analyze_noise_characteristics(data, dt):
    """
    Perform comprehensive noise analysis.
    
    Parameters:
    -----------
    data : np.array
        Acceleration data (already bias-removed)
    dt : float
        Sampling period in seconds
        
    Returns:
    --------
    dict
        Dictionary containing all noise metrics
    """
    results = {}
    
    # Basic statistics
    results['mean'] = np.mean(data)
    results['std'] = np.std(data)
    results['variance'] = np.var(data)
    results['rms'] = np.sqrt(np.mean(data**2))
    
    # Range statistics
    results['min'] = np.min(data)
    results['max'] = np.max(data)
    results['peak_to_peak'] = results['max'] - results['min']
    
    # Distribution statistics
    results['skewness'] = stats.skew(data)
    results['kurtosis'] = stats.kurtosis(data)
    
    # Quantiles
    results['percentile_25'] = np.percentile(data, 25)
    results['percentile_50'] = np.percentile(data, 50)  # Median
    results['percentile_75'] = np.percentile(data, 75)
    results['percentile_95'] = np.percentile(data, 95)
    results['percentile_99'] = np.percentile(data, 99)
    
    # Noise density
    results['noise_density'] = results['std'] / np.sqrt(1.0 / (2.0 * dt))  # m/s²/√Hz
    
    # Autocorrelation at lag 1
    if len(data) > 1:
        autocorr = np.correlate(data - results['mean'], data - results['mean'], mode='full')
        autocorr = autocorr / autocorr[len(autocorr)//2]  # Normalize
        results['autocorr_lag1'] = autocorr[len(autocorr)//2 + 1]
    
    return results


def calculate_power_spectral_density(data, dt):
    """
    Calculate Power Spectral Density.
    
    Parameters:
    -----------
    data : np.array
        Time series data
    dt : float
        Sampling period in seconds
        
    Returns:
    --------
    freq : np.array
        Frequency array
    psd : np.array
        Power spectral density
    """
    # Calculate PSD using Welch's method
    fs = 1.0 / dt  # Sampling frequency
    freq, psd = signal.welch(data, fs=fs, nperseg=min(256, len(data)//4))
    
    return freq, psd


def identify_noise_types(tau, adev):
    """
    Identify noise types from Allan deviation slope.
    
    Parameters:
    -----------
    tau : np.array
        Averaging times
    adev : np.array
        Allan deviation values
        
    Returns:
    --------
    dict
        Identified noise parameters
    """
    results = {}
    
    # Log-log fit to identify slope
    log_tau = np.log10(tau)
    log_adev = np.log10(adev)
    
    # Fit different regions
    if len(tau) > 10:
        # Short-term region (white noise)
        short_idx = len(tau) // 3
        slope_short, intercept_short = np.polyfit(log_tau[:short_idx], log_adev[:short_idx], 1)
        
        # Long-term region (bias instability/random walk)
        long_idx = 2 * len(tau) // 3
        slope_long, intercept_long = np.polyfit(log_tau[long_idx:], log_adev[long_idx:], 1)
        
        results['slope_short_term'] = slope_short
        results['slope_long_term'] = slope_long
        
        # Identify noise types based on slope
        # Slope -0.5: White noise
        # Slope 0: Bias instability
        # Slope 0.5: Random walk
        
        if abs(slope_short + 0.5) < 0.2:
            results['dominant_short_term'] = "White noise"
            # White noise coefficient (at tau=1)
            results['white_noise_coeff'] = 10**intercept_short
        else:
            results['dominant_short_term'] = f"Unknown (slope={slope_short:.2f})"
        
        if abs(slope_long) < 0.2:
            results['dominant_long_term'] = "Bias instability"
            # Find minimum of Allan deviation (bias instability)
            min_idx = np.argmin(adev)
            results['bias_instability'] = adev[min_idx]
            results['bias_instability_time'] = tau[min_idx]
        elif abs(slope_long - 0.5) < 0.2:
            results['dominant_long_term'] = "Random walk"
            # Random walk coefficient
            results['random_walk_coeff'] = 10**(intercept_long - 0.5)
        else:
            results['dominant_long_term'] = f"Unknown (slope={slope_long:.2f})"
    
    return results


def plot_noise_analysis(acc_x, acc_y, dt, save_path='noise_analysis.png'):
    """
    Create comprehensive noise analysis plots.
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Calculate Allan variance for both axes
    tau_x, _, adev_x = allan_variance(acc_x, dt)
    tau_y, _, adev_y = allan_variance(acc_y, dt)
    
    # Calculate PSD
    freq_x, psd_x = calculate_power_spectral_density(acc_x, dt)
    freq_y, psd_y = calculate_power_spectral_density(acc_y, dt)
    
    # Plot 1: Time series
    ax1 = plt.subplot(3, 3, 1)
    time = np.arange(len(acc_x)) * dt
    ax1.plot(time[:1000], acc_x[:1000], 'b-', alpha=0.5, linewidth=0.5, label='X')
    ax1.plot(time[:1000], acc_y[:1000], 'r-', alpha=0.5, linewidth=0.5, label='Y')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Acceleration (m/s²)')
    ax1.set_title('Time Series (first 1000 samples)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Allan Deviation
    ax2 = plt.subplot(3, 3, 2)
    ax2.loglog(tau_x, adev_x, 'b-', label='X axis')
    ax2.loglog(tau_y, adev_y, 'r-', label='Y axis')
    ax2.set_xlabel('Averaging Time τ (s)')
    ax2.set_ylabel('Allan Deviation (m/s²)')
    ax2.set_title('Allan Deviation')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    # Plot 3: Power Spectral Density
    ax3 = plt.subplot(3, 3, 3)
    ax3.semilogy(freq_x, psd_x, 'b-', alpha=0.7, label='X axis')
    ax3.semilogy(freq_y, psd_y, 'r-', alpha=0.7, label='Y axis')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('PSD (m²/s⁴/Hz)')
    ax3.set_title('Power Spectral Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Histogram X
    ax4 = plt.subplot(3, 3, 4)
    n, bins, _ = ax4.hist(acc_x, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    # Fit normal distribution
    mu, sigma = np.mean(acc_x), np.std(acc_x)
    x_fit = np.linspace(acc_x.min(), acc_x.max(), 100)
    ax4.plot(x_fit, stats.norm.pdf(x_fit, mu, sigma), 'r-', linewidth=2, label=f'Normal fit\nμ={mu:.4f}\nσ={sigma:.4f}')
    ax4.set_xlabel('Acceleration (m/s²)')
    ax4.set_ylabel('Probability Density')
    ax4.set_title('X-axis Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Histogram Y
    ax5 = plt.subplot(3, 3, 5)
    n, bins, _ = ax5.hist(acc_y, bins=50, density=True, alpha=0.7, color='red', edgecolor='black')
    mu, sigma = np.mean(acc_y), np.std(acc_y)
    y_fit = np.linspace(acc_y.min(), acc_y.max(), 100)
    ax5.plot(y_fit, stats.norm.pdf(y_fit, mu, sigma), 'b-', linewidth=2, label=f'Normal fit\nμ={mu:.4f}\nσ={sigma:.4f}')
    ax5.set_xlabel('Acceleration (m/s²)')
    ax5.set_ylabel('Probability Density')
    ax5.set_title('Y-axis Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Q-Q plot for normality test
    ax6 = plt.subplot(3, 3, 6)
    stats.probplot(acc_x, dist="norm", plot=ax6)
    ax6.set_title('Q-Q Plot (X-axis)')
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Autocorrelation X
    ax7 = plt.subplot(3, 3, 7)
    lags = np.arange(0, min(100, len(acc_x)//10))
    autocorr_x = [np.corrcoef(acc_x[:-lag if lag > 0 else len(acc_x):], acc_x[lag:])[0, 1] if lag < len(acc_x) else 0 for lag in lags]
    ax7.stem(lags * dt, autocorr_x, basefmt=' ')
    ax7.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax7.axhline(y=0.05, color='r', linestyle='--', alpha=0.5)
    ax7.axhline(y=-0.05, color='r', linestyle='--', alpha=0.5)
    ax7.set_xlabel('Lag (s)')
    ax7.set_ylabel('Autocorrelation')
    ax7.set_title('Autocorrelation (X-axis)')
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Autocorrelation Y
    ax8 = plt.subplot(3, 3, 8)
    autocorr_y = [np.corrcoef(acc_y[:-lag if lag > 0 else len(acc_y):], acc_y[lag:])[0, 1] if lag < len(acc_y) else 0 for lag in lags]
    ax8.stem(lags * dt, autocorr_y, basefmt=' ', markerfmt='ro')
    ax8.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax8.axhline(y=0.05, color='r', linestyle='--', alpha=0.5)
    ax8.axhline(y=-0.05, color='r', linestyle='--', alpha=0.5)
    ax8.set_xlabel('Lag (s)')
    ax8.set_ylabel('Autocorrelation')
    ax8.set_title('Autocorrelation (Y-axis)')
    ax8.grid(True, alpha=0.3)
    
    # Plot 9: Cumulative sum (drift check)
    ax9 = plt.subplot(3, 3, 9)
    cumsum_x = np.cumsum(acc_x) * dt
    cumsum_y = np.cumsum(acc_y) * dt
    ax9.plot(time, cumsum_x, 'b-', alpha=0.7, label='X axis')
    ax9.plot(time, cumsum_y, 'r-', alpha=0.7, label='Y axis')
    ax9.set_xlabel('Time (s)')
    ax9.set_ylabel('Cumulative Sum (m/s)')
    ax9.set_title('Integrated Acceleration (Velocity Drift)')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle('IMU Noise Analysis - Stationary Data', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Noise analysis plot saved to {save_path}")


def main():
    """Main function for noise analysis."""
    from read_imu_data import read_imu_data
    
    print("=" * 60)
    print("IMU NOISE ANALYSIS")
    print("=" * 60)
    
    # Read statistics file
    print("\nReading statistics.txt (stationary data)...")
    data = read_imu_data('data/statistics.txt')
    print(f"Loaded {len(data)} samples")
    
    # Get time step
    timestamps = data['timestamp'].values
    dt = np.median(np.diff(timestamps)) / 1000.0  # Convert to seconds
    print(f"Sampling period: {dt*1000:.2f} ms ({1/dt:.1f} Hz)")
    
    # Get accelerometer data
    acc_x = data['x'].values
    acc_y = data['y'].values
    
    # Remove bias (mean)
    bias_x = np.mean(acc_x)
    bias_y = np.mean(acc_y)
    acc_x_centered = acc_x - bias_x
    acc_y_centered = acc_y - bias_y
    
    print("\n" + "=" * 60)
    print("BIAS ANALYSIS")
    print("=" * 60)
    print(f"X-axis bias: {bias_x:.6f} m/s²")
    print(f"Y-axis bias: {bias_y:.6f} m/s²")
    print(f"Magnitude bias: {np.sqrt(bias_x**2 + bias_y**2):.6f} m/s²")
    
    # Analyze noise characteristics
    print("\n" + "=" * 60)
    print("NOISE STATISTICS (bias removed)")
    print("=" * 60)
    
    print("\nX-axis noise:")
    noise_x = analyze_noise_characteristics(acc_x_centered, dt)
    for key, value in noise_x.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:.6f}")
    
    print("\nY-axis noise:")
    noise_y = analyze_noise_characteristics(acc_y_centered, dt)
    for key, value in noise_y.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:.6f}")
    
    # Allan variance analysis
    print("\n" + "=" * 60)
    print("ALLAN VARIANCE ANALYSIS")
    print("=" * 60)
    
    print("\nCalculating Allan variance...")
    tau_x, avar_x, adev_x = allan_variance(acc_x_centered, dt)
    tau_y, avar_y, adev_y = allan_variance(acc_y_centered, dt)
    
    # Identify noise types
    print("\nX-axis noise identification:")
    noise_types_x = identify_noise_types(tau_x, adev_x)
    for key, value in noise_types_x.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    print("\nY-axis noise identification:")
    noise_types_y = identify_noise_types(tau_y, adev_y)
    for key, value in noise_types_y.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # Power spectral density
    print("\n" + "=" * 60)
    print("POWER SPECTRAL DENSITY")
    print("=" * 60)
    
    freq_x, psd_x = calculate_power_spectral_density(acc_x_centered, dt)
    freq_y, psd_y = calculate_power_spectral_density(acc_y_centered, dt)
    
    # Find dominant frequencies
    peak_freq_x = freq_x[np.argmax(psd_x[1:])+1]  # Skip DC
    peak_freq_y = freq_y[np.argmax(psd_y[1:])+1]
    
    print(f"\nX-axis peak frequency: {peak_freq_x:.2f} Hz")
    print(f"Y-axis peak frequency: {peak_freq_y:.2f} Hz")
    
    # Noise density at 1 Hz
    idx_1hz = np.argmin(np.abs(freq_x - 1.0))
    print(f"\nNoise density at 1 Hz:")
    print(f"  X-axis: {np.sqrt(psd_x[idx_1hz]):.6e} m/s²/√Hz")
    print(f"  Y-axis: {np.sqrt(psd_y[idx_1hz]):.6e} m/s²/√Hz")
    
    # Statistical tests
    print("\n" + "=" * 60)
    print("STATISTICAL TESTS")
    print("=" * 60)
    
    # Normality test (Shapiro-Wilk)
    if len(acc_x_centered) > 5000:
        # Use subset for Shapiro test (max 5000 samples)
        sample_x = np.random.choice(acc_x_centered, 5000, replace=False)
        sample_y = np.random.choice(acc_y_centered, 5000, replace=False)
    else:
        sample_x = acc_x_centered
        sample_y = acc_y_centered
    
    stat_x, p_x = stats.shapiro(sample_x)
    stat_y, p_y = stats.shapiro(sample_y)
    
    print(f"\nShapiro-Wilk normality test:")
    print(f"  X-axis: statistic={stat_x:.4f}, p-value={p_x:.4e}")
    print(f"    {'Normal' if p_x > 0.05 else 'Non-normal'} distribution (α=0.05)")
    print(f"  Y-axis: statistic={stat_y:.4f}, p-value={p_y:.4e}")
    print(f"    {'Normal' if p_y > 0.05 else 'Non-normal'} distribution (α=0.05)")
    
    # White noise test (Ljung-Box)
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_x = acorr_ljungbox(acc_x_centered, lags=10, return_df=True)
    lb_y = acorr_ljungbox(acc_y_centered, lags=10, return_df=True)
    
    print(f"\nLjung-Box white noise test (lag=10):")
    print(f"  X-axis p-value: {lb_x['lb_pvalue'].iloc[-1]:.4e}")
    print(f"    {'White noise' if lb_x['lb_pvalue'].iloc[-1] > 0.05 else 'Correlated'} (α=0.05)")
    print(f"  Y-axis p-value: {lb_y['lb_pvalue'].iloc[-1]:.4e}")
    print(f"    {'White noise' if lb_y['lb_pvalue'].iloc[-1] > 0.05 else 'Correlated'} (α=0.05)")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - KEY NOISE PARAMETERS")
    print("=" * 60)
    print(f"\nSampling: {1/dt:.1f} Hz")
    print(f"\nBias:")
    print(f"  X: {bias_x:.6f} m/s²")
    print(f"  Y: {bias_y:.6f} m/s²")
    print(f"\nWhite noise (1σ):")
    print(f"  X: {noise_x['std']:.6f} m/s²")
    print(f"  Y: {noise_y['std']:.6f} m/s²")
    print(f"\nNoise density:")
    print(f"  X: {noise_x['noise_density']:.6e} m/s²/√Hz")
    print(f"  Y: {noise_y['noise_density']:.6e} m/s²/√Hz")
    
    if 'bias_instability' in noise_types_x:
        print(f"\nBias instability:")
        print(f"  X: {noise_types_x['bias_instability']:.6e} m/s² at τ={noise_types_x['bias_instability_time']:.1f}s")
    if 'bias_instability' in noise_types_y:
        print(f"  Y: {noise_types_y['bias_instability']:.6e} m/s² at τ={noise_types_y['bias_instability_time']:.1f}s")
    
    # Generate plots
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    plot_noise_analysis(acc_x_centered, acc_y_centered, dt, 'imu_noise_analysis.png')
    
    # Save results to file
    results = {
        'sampling_rate_hz': 1/dt,
        'samples': len(data),
        'duration_s': len(data) * dt,
        'bias_x': bias_x,
        'bias_y': bias_y,
        'noise_x': noise_x,
        'noise_y': noise_y,
        'allan_x': {
            'tau': tau_x.tolist(),
            'allan_dev': adev_x.tolist()
        },
        'allan_y': {
            'tau': tau_y.tolist(),
            'allan_dev': adev_y.tolist()
        }
    }
    
    import json
    with open('noise_statistics.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("Noise statistics saved to noise_statistics.json")


if __name__ == "__main__":
    main()