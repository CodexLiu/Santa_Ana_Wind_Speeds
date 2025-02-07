import numpy as np
import pandas as pd
from scipy import stats
from utils import (
    generate_timestamps,
    add_temporal_correlation,
    add_seasonal_pattern,
    transform_to_uniform
)


class WeatherDataGenerator:
    def __init__(self, start_year=1940, end_year=2025):
        """Initialize the weather data generator.

        Args:
            start_year (int): Starting year for simulation
            end_year (int): Ending year for simulation
        """
        self.timestamps = generate_timestamps(start_year, end_year)
        self.n_samples = len(self.timestamps)

    def generate_wind_speed(self, station_id, mean=10, scale=5):
        """Generate wind speed data following Weibull distribution.

        Args:
            station_id (int): Weather station identifier
            mean (float): Mean wind speed
            scale (float): Scale parameter for Weibull distribution

        Returns:
            np.array: Generated wind speed data
        """
        # Weibull distribution parameters
        shape = 2.0  # typical for wind speeds

        # Generate base distribution
        raw_data = stats.weibull_min.rvs(
            shape, loc=0, scale=scale, size=self.n_samples
        )

        # Add temporal correlation
        corr_data = add_temporal_correlation(raw_data, correlation_length=24)

        # Add seasonal pattern (stronger winds in winter)
        seasonal = add_seasonal_pattern(
            self.timestamps,
            amplitude=2.0,
            phase_shift=np.pi  # peak in winter
        )

        return corr_data + seasonal + mean

    def generate_temperature(self, mean=20, amplitude=10):
        """Generate temperature data with seasonal patterns.

        Args:
            mean (float): Mean temperature
            amplitude (float): Seasonal amplitude

        Returns:
            np.array: Generated temperature data
        """
        # Generate seasonal pattern
        seasonal = add_seasonal_pattern(
            self.timestamps,
            amplitude=amplitude,
            phase_shift=0  # peak in summer
        )

        # Add random variations (normal distribution)
        noise = np.random.normal(0, 2, size=self.n_samples)
        corr_noise = add_temporal_correlation(noise, correlation_length=48)

        return seasonal + mean + corr_noise

    def generate_soil_moisture(self, mean=0.3, min_val=0.1, max_val=0.5):
        """Generate soil moisture data (bounded between 0 and 1).

        Args:
            mean (float): Mean soil moisture
            min_val (float): Minimum soil moisture
            max_val (float): Maximum soil moisture

        Returns:
            np.array: Generated soil moisture data
        """
        # Generate base data using beta distribution
        alpha = 2.0
        beta = 2.0
        raw_data = stats.beta.rvs(alpha, beta, size=self.n_samples)

        # Add temporal correlation
        corr_data = add_temporal_correlation(raw_data, correlation_length=72)

        # Scale to desired range
        scaled_data = (max_val - min_val) * corr_data + min_val

        # Add seasonal pattern (higher in winter)
        seasonal = add_seasonal_pattern(
            self.timestamps,
            amplitude=0.1,
            phase_shift=np.pi  # peak in winter
        )

        return np.clip(scaled_data + seasonal, min_val, max_val)

    def generate_dataset(self):
        """Generate complete dataset with all variables.

        Returns:
            pd.DataFrame: Generated weather dataset
        """
        data = {
            'timestamp': self.timestamps,
            'wind_speed_1': self.generate_wind_speed(station_id=1),
            'wind_speed_2': self.generate_wind_speed(station_id=2),
            'temperature': self.generate_temperature(),
            'soil_moisture': self.generate_soil_moisture()
        }

        return pd.DataFrame(data).set_index('timestamp')


if __name__ == "__main__":
    # Generate sample dataset
    generator = WeatherDataGenerator()
    data = generator.generate_dataset()

    # Save to CSV
    data.to_csv('synthetic_weather_data.csv')

    print("Generated synthetic weather data with shape:", data.shape)
    print("\nSample of the generated data:")
    print(data.head())
