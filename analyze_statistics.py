#!/usr/bin/env python3
"""
Script to analyze statistics.txt and calculate covariance for x and y axes.
"""

import pandas as pd
import numpy as np


def read_statistics_data(filepath='data/statistics.txt'):
    """
    Read statistics data from the txt file.
    
    Parameters:
    -----------
    filepath : str
        Path to the statistics file
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing timestamp, x and y acceleration data
    """
    # Read the data file
    data = pd.read_csv(filepath, sep='\t', header=None, 
                       names=['timestamp', 'data_type', 'x', 'y', 'z'])
    
    # Filter only ACC data
    acc_data = data[data['data_type'] == 'ACC'].copy()
    
    # Keep only timestamp, x, and y columns
    acc_data = acc_data[['timestamp', 'x', 'y']]
    
    # Reset index
    acc_data.reset_index(drop=True, inplace=True)
    
    return acc_data


def calculate_covariance_analysis(data):
    """
    Calculate covariance analysis for x and y axes.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with x and y columns
        
    Returns:
    --------
    dict
        Dictionary containing covariance analysis results
    """
    results = {}
    
    # Calculate means
    results['x_mean'] = data['x'].mean()
    results['y_mean'] = data['y'].mean()
    
    # Calculate variances (diagonal elements of covariance matrix)
    results['x_variance'] = data['x'].var()
    results['y_variance'] = data['y'].var()
    
    # Calculate standard deviations
    results['x_std'] = data['x'].std()
    results['y_std'] = data['y'].std()
    
    # Calculate covariance between x and y
    results['xy_covariance'] = data['x'].cov(data['y'])
    
    # Calculate full covariance matrix
    cov_matrix = data[['x', 'y']].cov()
    results['covariance_matrix'] = cov_matrix
    
    # Calculate correlation coefficient
    results['correlation'] = data['x'].corr(data['y'])
    
    return results


def main():
    """Main function to analyze statistics data."""
    
    print("Reading statistics.txt file...")
    print("=" * 60)
    
    # Read the data
    stats_data = read_statistics_data('data/statistics.txt')
    
    print(f"Loaded {len(stats_data)} samples")
    print(f"Time range: {stats_data['timestamp'].min()} to {stats_data['timestamp'].max()}")
    print(f"Duration: {(stats_data['timestamp'].max() - stats_data['timestamp'].min()) / 1000:.2f} seconds")
    
    # Calculate covariance analysis
    print("\n" + "=" * 60)
    print("COVARIANCE ANALYSIS")
    print("=" * 60)
    
    results = calculate_covariance_analysis(stats_data)
    
    print("\n1. Mean values:")
    print(f"   X-axis mean: {results['x_mean']:.6f}")
    print(f"   Y-axis mean: {results['y_mean']:.6f}")
    
    print("\n2. Variance (diagonal elements of covariance matrix):")
    print(f"   X-axis variance: {results['x_variance']:.8f}")
    print(f"   Y-axis variance: {results['y_variance']:.8f}")
    
    print("\n3. Standard deviation:")
    print(f"   X-axis std dev: {results['x_std']:.6f}")
    print(f"   Y-axis std dev: {results['y_std']:.6f}")
    
    print("\n4. Covariance between X and Y:")
    print(f"   Cov(X,Y): {results['xy_covariance']:.8f}")
    
    print("\n5. Full Covariance Matrix:")
    print("   [[Var(X)    Cov(X,Y)]")
    print("    [Cov(Y,X)  Var(Y)  ]]")
    print("\n   Numerical values:")
    print(results['covariance_matrix'])
    
    print("\n6. Correlation coefficient:")
    print(f"   Correlation(X,Y): {results['correlation']:.6f}")
    
    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    
    if abs(results['correlation']) < 0.3:
        correlation_strength = "weak"
    elif abs(results['correlation']) < 0.7:
        correlation_strength = "moderate"
    else:
        correlation_strength = "strong"
    
    if results['correlation'] > 0:
        correlation_direction = "positive"
    else:
        correlation_direction = "negative"
    
    print(f"\nThe X and Y axes show a {correlation_strength} {correlation_direction} correlation.")
    print(f"The covariance value of {results['xy_covariance']:.8f} indicates that the two axes")
    print(f"{'move together' if results['xy_covariance'] > 0 else 'move in opposite directions'}.")
    
    # Additional statistics
    print("\n" + "=" * 60)
    print("ADDITIONAL STATISTICS")
    print("=" * 60)
    
    print("\nData range:")
    print(f"   X-axis: [{stats_data['x'].min():.6f}, {stats_data['x'].max():.6f}]")
    print(f"   Y-axis: [{stats_data['y'].min():.6f}, {stats_data['y'].max():.6f}]")
    
    print("\nPercentiles:")
    percentiles = [25, 50, 75]
    for p in percentiles:
        x_p = stats_data['x'].quantile(p/100)
        y_p = stats_data['y'].quantile(p/100)
        print(f"   {p}th percentile - X: {x_p:.6f}, Y: {y_p:.6f}")


if __name__ == "__main__":
    main()