import numpy as np
import pandas as pd
from scipy import stats
from copulas.multivariate import GaussianMultivariate
from utils import transform_to_uniform, calculate_temporal_acf
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os


class WeatherDistributionFitter:
    def __init__(self, data):
        """Initialize the distribution fitter.

        Args:
            data (pd.DataFrame): Weather data with columns for wind speeds,
                               temperature, and soil moisture
        """
        self.data = data
        self.fitted_distributions = {}
        self.copula_model = None

        # Create plots directory if it doesn't exist
        self.plots_dir = 'plots'
        os.makedirs(self.plots_dir, exist_ok=True)

        # Set modern style
        sns.set_theme(style="whitegrid")
        sns.set_palette("husl")

    def fit_wind_speed_distribution(self, station_col):
        """Fit Weibull distribution to wind speed data.

        Args:
            station_col (str): Column name for wind speed data

        Returns:
            tuple: Fitted distribution parameters
        """
        data = self.data[station_col].values
        # Fit Weibull distribution using MLE
        shape, loc, scale = stats.weibull_min.fit(data)
        self.fitted_distributions[station_col] = {
            'distribution': 'weibull_min',
            'parameters': {'shape': shape, 'loc': loc, 'scale': scale}
        }
        return shape, loc, scale

    def fit_temperature_distribution(self):
        """Fit normal distribution to deseasonalized temperature data.

        Returns:
            tuple: Fitted distribution parameters
        """
        data = self.data['temperature'].values
        # Remove seasonal component (simple moving average)
        window = 24 * 30  # 30-day window
        ma = pd.Series(data).rolling(window=window, center=True).mean()
        deseasonalized = data - \
            ma.ffill().bfill()  # Using ffill/bfill instead of fillna method

        # Fit normal distribution
        loc, scale = stats.norm.fit(deseasonalized)
        self.fitted_distributions['temperature'] = {
            'distribution': 'norm',
            'parameters': {'loc': loc, 'scale': scale}
        }
        return loc, scale

    def fit_soil_moisture_distribution(self):
        """Fit beta distribution to soil moisture data.

        Returns:
            tuple: Fitted distribution parameters
        """
        data = self.data['soil_moisture'].values
        # Scale data to [0,1] range for beta distribution
        scaled_data = (data - data.min()) / (data.max() - data.min())

        # Fit beta distribution
        alpha, beta, loc, scale = stats.beta.fit(scaled_data)
        self.fitted_distributions['soil_moisture'] = {
            'distribution': 'beta',
            'parameters': {'alpha': alpha, 'beta': beta, 'loc': loc, 'scale': scale}
        }
        return alpha, beta, loc, scale

    def fit_all_marginals(self):
        """Fit distributions to all variables."""
        self.fit_wind_speed_distribution('wind_speed_1')
        self.fit_wind_speed_distribution('wind_speed_2')
        self.fit_temperature_distribution()
        self.fit_soil_moisture_distribution()

    def transform_to_uniform_margins(self):
        """Transform all variables to uniform margins.

        Returns:
            pd.DataFrame: Data with uniform margins
        """
        uniform_data = pd.DataFrame(index=self.data.index)

        for col in self.data.columns:
            uniform_data[col] = transform_to_uniform(self.data[col].values)

        return uniform_data

    def fit_copula(self):
        """Fit Gaussian copula to transformed data.

        Returns:
            GaussianMultivariate: Fitted copula model
        """
        uniform_data = self.transform_to_uniform_margins()
        self.copula_model = GaussianMultivariate()
        self.copula_model.fit(uniform_data)
        return self.copula_model

    def analyze_temporal_dependence(self, max_lag=48):
        """Analyze temporal dependence structure.

        Args:
            max_lag (int): Maximum lag for autocorrelation

        Returns:
            dict: Autocorrelation functions for each variable
        """
        acf_results = {}
        for col in self.data.columns:
            acf_results[col] = calculate_temporal_acf(
                self.data[col].values,
                max_lag=max_lag
            )
        return acf_results

    def plot_wind_speed_distribution(self, station_col):
        """Plot and save wind speed distribution with fitted Weibull."""
        plt.figure(figsize=(10, 6))
        data = self.data[station_col].values
        dist_params = self.fitted_distributions[station_col]['parameters']

        sns.histplot(data, stat='density', alpha=0.6, label='Observed')
        x = np.linspace(data.min(), data.max(), 100)
        pdf = stats.weibull_min.pdf(x, dist_params['shape'],
                                    dist_params['loc'],
                                    dist_params['scale'])
        plt.plot(x, pdf, 'r-', lw=2, label='Fitted Weibull')

        plt.title(f'Wind Speed Distribution - {station_col}')
        plt.xlabel('Wind Speed')
        plt.ylabel('Density')
        plt.legend()

        # Save plot
        plt.savefig(os.path.join(self.plots_dir, f'{station_col}_distribution.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_temperature_distribution(self):
        """Plot and save temperature distribution with fitted normal."""
        plt.figure(figsize=(10, 6))
        data = self.data['temperature'].values
        dist_params = self.fitted_distributions['temperature']['parameters']

        sns.histplot(data, stat='density', alpha=0.6, label='Observed')
        x = np.linspace(data.min(), data.max(), 100)
        pdf = stats.norm.pdf(x, dist_params['loc'], dist_params['scale'])
        plt.plot(x, pdf, 'r-', lw=2, label='Fitted Normal')

        plt.title('Temperature Distribution')
        plt.xlabel('Temperature')
        plt.ylabel('Density')
        plt.legend()

        plt.savefig(os.path.join(self.plots_dir, 'temperature_distribution.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_soil_moisture_distribution(self):
        """Plot and save soil moisture distribution with fitted beta."""
        plt.figure(figsize=(10, 6))
        data = self.data['soil_moisture'].values
        dist_params = self.fitted_distributions['soil_moisture']['parameters']

        sns.histplot(data, stat='density', alpha=0.6, label='Observed')
        x = np.linspace(data.min(), data.max(), 100)
        pdf = stats.beta.pdf(x, dist_params['alpha'],
                             dist_params['beta'],
                             dist_params['loc'],
                             dist_params['scale'])
        plt.plot(x, pdf, 'r-', lw=2, label='Fitted Beta')

        plt.title('Soil Moisture Distribution')
        plt.xlabel('Soil Moisture')
        plt.ylabel('Density')
        plt.legend()

        plt.savefig(os.path.join(self.plots_dir, 'soil_moisture_distribution.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_temporal_patterns(self, days=30):
        """Plot and save temporal patterns for all variables.

        Args:
            days (int): Number of days to plot
        """
        hours = days * 24
        data_subset = self.data.iloc[:hours]

        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 1, figure=fig)

        # Plot wind speeds
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(data_subset.index, data_subset['wind_speed_1'],
                 label='Station 1', alpha=0.7)
        ax1.plot(data_subset.index, data_subset['wind_speed_2'],
                 label='Station 2', alpha=0.7)
        ax1.set_title('Wind Speeds')
        ax1.legend()

        # Plot temperature
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(data_subset.index, data_subset['temperature'],
                 color='red', alpha=0.7)
        ax2.set_title('Temperature')

        # Plot soil moisture
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(data_subset.index, data_subset['soil_moisture'],
                 color='brown', alpha=0.7)
        ax3.set_title('Soil Moisture')

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'temporal_patterns.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_acf(self, max_lag=48):
        """Plot and save autocorrelation functions for all variables."""
        acf_results = self.analyze_temporal_dependence(max_lag)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()

        for idx, (var, acf) in enumerate(acf_results.items()):
            axes[idx].bar(range(len(acf)), acf, alpha=0.6)
            axes[idx].axhline(y=0, color='r', linestyle='-', alpha=0.2)
            axes[idx].axhline(y=1.96/np.sqrt(len(self.data)),
                              color='r', linestyle='--', alpha=0.2)
            axes[idx].axhline(y=-1.96/np.sqrt(len(self.data)),
                              color='r', linestyle='--', alpha=0.2)
            axes[idx].set_title(f'ACF - {var}')
            axes[idx].set_xlabel('Lag')
            axes[idx].set_ylabel('ACF')

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'autocorrelation.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_copula_dependencies(self):
        """Plot and save pairwise dependencies using scatter plots."""
        uniform_data = self.transform_to_uniform_margins()

        fig = plt.figure(figsize=(12, 12))
        gs = GridSpec(4, 4, figure=fig)

        variables = list(uniform_data.columns)
        n_vars = len(variables)

        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    ax = fig.add_subplot(gs[i, j])
                    ax.scatter(uniform_data[variables[j]],
                               uniform_data[variables[i]],
                               alpha=0.1, s=1)
                    if i == n_vars-1:
                        ax.set_xlabel(variables[j])
                    if j == 0:
                        ax.set_ylabel(variables[i])
                else:
                    ax = fig.add_subplot(gs[i, i])
                    ax.text(0.5, 0.5, variables[i],
                            ha='center', va='center')
                    ax.axis('off')

        plt.suptitle('Copula Dependencies (Uniform Margins)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'copula_dependencies.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_all(self):
        """Generate and save all plots."""
        print("Generating plots...")
        self.plot_wind_speed_distribution('wind_speed_1')
        self.plot_wind_speed_distribution('wind_speed_2')
        self.plot_temperature_distribution()
        self.plot_soil_moisture_distribution()
        self.plot_temporal_patterns()
        self.plot_acf()
        self.plot_copula_dependencies()
        print(f"All plots saved to '{self.plots_dir}' directory")


if __name__ == "__main__":
    # Load synthetic data
    print("Loading data...")
    data = pd.read_csv('synthetic_weather_data.csv', index_col='timestamp')
    data.index = pd.to_datetime(data.index)

    # Fit distributions
    print("Fitting distributions...")
    fitter = WeatherDistributionFitter(data)
    fitter.fit_all_marginals()

    # Fit copula
    print("Fitting copula...")
    copula_model = fitter.fit_copula()

    # Generate and save all plots
    fitter.plot_all()

    print("\nFitted distributions:")
    for var, dist_info in fitter.fitted_distributions.items():
        print(f"\n{var}:")
        print(f"Distribution: {dist_info['distribution']}")
        print(f"Parameters: {dist_info['parameters']}")
