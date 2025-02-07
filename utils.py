import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta


def generate_timestamps(start_year=1940, end_year=2025, freq='H'):
    """Generate timestamps for the simulation period.

    Args:
        start_year (int): Starting year for simulation
        end_year (int): Ending year for simulation
        freq (str): Frequency of timestamps ('H' for hourly)

    Returns:
        pd.DatetimeIndex: Timestamps for the simulation period
    """
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    return pd.date_range(start=start_date, end=end_date, freq=freq)


def add_temporal_correlation(data, correlation_length=24):
    """Add temporal correlation to a time series using exponential smoothing.

    Args:
        data (np.array): Original time series data
        correlation_length (int): Number of time steps for correlation

    Returns:
        np.array: Temporally correlated time series
    """
    kernel = np.exp(-np.arange(correlation_length) / (correlation_length / 3))
    kernel = kernel / kernel.sum()
    return np.convolve(data, kernel, mode='same')


def add_seasonal_pattern(timestamps, amplitude=1.0, phase_shift=0):
    """Generate seasonal pattern for a time series.

    Args:
        timestamps (pd.DatetimeIndex): Time stamps
        amplitude (float): Amplitude of seasonal variation
        phase_shift (float): Phase shift in radians

    Returns:
        np.array: Seasonal pattern
    """
    # Convert timestamps to radians (2π per year)
    time_radians = 2 * np.pi * (timestamps.dayofyear / 365.25)
    return amplitude * np.sin(time_radians + phase_shift)


def transform_to_uniform(data):
    """Transform data to uniform distribution using empirical CDF.

    Args:
        data (np.array): Input data

    Returns:
        np.array: Transformed data with uniform distribution
    """
    return stats.rankdata(data) / (len(data) + 1)


def inverse_transform(uniform_data, distribution):
    """Transform uniform data back to original distribution.

    Args:
        uniform_data (np.array): Uniform distributed data
        distribution: scipy.stats distribution object

    Returns:
        np.array: Transformed data following specified distribution
    """
    return distribution.ppf(uniform_data)


def calculate_temporal_acf(data, max_lag=48):
    """Calculate temporal autocorrelation function.

    Args:
        data (np.array): Time series data
        max_lag (int): Maximum lag to calculate

    Returns:
        np.array: Autocorrelation values for different lags
    """
    mean = np.mean(data)
    variance = np.var(data)
    normalized_data = data - mean
    acf = np.correlate(normalized_data, normalized_data,
                       mode='full')[len(data)-1:]
    acf = acf[:max_lag] / (variance * len(data))
    return acf
