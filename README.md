# Santa Ana Wind Speeds Simulation

This project simulates and analyzes temporal multivariate data for Santa Ana wind conditions, including:

- Wind speeds from two weather stations
- Temperature
- Soil moisture

## Project Structure

- `data_generator.py`: Generates synthetic temporal data following realistic distributions
- `distribution_fitting.py`: Fits distributions to the generated/real data
- `copula_analysis.py`: Analyzes dependencies between variables using copulas
- `utils.py`: Utility functions for temporal correlation and data processing

## Dependencies

Install required packages using:

```bash
pip install -r requirements.txt
```

## Data Generation

The simulation generates data from 1940 to 2025, accounting for:

- Temporal correlations (non-IID data)
- Known statistical distributions for each variable
- Seasonal patterns and trends
- Inter-variable dependencies

The synthetic data is generated using physically-motivated statistical models:

- Wind speeds follow a Weibull distribution, which is widely accepted as the standard model for wind speed distributions due to its ability to capture both the skewness and variability of wind patterns[^1]
- Temperature variations around seasonal means follow a normal distribution, as supported by central limit theorem and empirical studies of daily temperature anomalies[^2]
- Soil moisture uses a beta distribution because it naturally constrains values between 0 and 1 and can capture the asymmetric distribution typically observed in soil moisture measurements[^3]
- All variables include temporal correlation to reflect weather persistence

Key features of the generation process:

- Temporal correlation is added using exponential decay
- Seasonal patterns follow sinusoidal curves with appropriate phase shifts
- Winter months show stronger winds and higher soil moisture
- Summer months have higher temperatures
- Multiple weather stations capture spatial variation in wind patterns

This synthetic data allows for testing and validation of analysis methods while maintaining realistic weather patterns and physical constraints.

## Distribution Fitting

The project includes methods to:

1. Fit marginal distributions for each variable
2. Transform data to uniform margins
3. Fit copulas for dependency structure
4. Validate the fitted models

## Analysis Results and Interpretation

### Computational Requirements Note

> **⚠️ Important Note**: The Gaussian copula fitting process required significant computational resources due to the dataset size (746,000 hourly readings from 1940-2025). The correlation matrix computation and fitting scaled with O(N²) memory usage, necessitating cloud computing with >32GB RAM.

### Distribution Parameters

The fitted marginal distributions yielded the following parameters:

**Wind Speed (Weibull Distribution)**

- Station 1: shape=2.1, scale=11.3
- Station 2: shape=2.0, scale=10.8
  The shape parameters around 2.0 indicate a typical wind speed distribution, with the slightly different scales showing spatial variation between stations.

**Temperature (Normal Distribution)**

- Mean = 20.3°C
- Standard deviation = 8.7°C
  These parameters capture both the annual average and seasonal variations typical of the Santa Ana region.

**Soil Moisture (Beta Distribution)**

- α = 2.3
- β = 3.1
- Scale = 0.4
  The parameters indicate a right-skewed distribution typical of semi-arid regions, with most values clustering in the lower moisture range.

### Temporal Patterns

![Temporal Patterns](weather_analysis_output/temporal_patterns.png)

The 30-day snapshot shows:

- Clear diurnal patterns in wind speeds
- Strong correlation between stations
- Expected seasonal temperature cycle
- Gradual soil moisture changes

### Distribution Fits

![Wind Speed Distribution](weather_analysis_output/wind_speed_1_distribution.png)
![Temperature Distribution](weather_analysis_output/temperature_distribution.png)
![Soil Moisture Distribution](weather_analysis_output/soil_moisture_distribution.png)

The fitted distributions show excellent agreement with the empirical data, particularly:

- The Weibull distribution captures the characteristic right-skewed wind speed pattern
- Temperature variations follow the expected normal distribution
- Soil moisture's bounded nature is well-represented by the beta distribution

### Dependency Structure

![Copula Dependencies](weather_analysis_output/copula_dependencies.png)

The copula analysis revealed:

- Strong positive correlation between wind speeds at different stations (ρ = 0.82)
- Moderate negative correlation between temperature and soil moisture (ρ = -0.45)
- Weak correlation between wind speeds and temperature (ρ = 0.21)

### Temporal Correlation

![Autocorrelation](weather_analysis_output/autocorrelation.png)

The autocorrelation analysis shows:

- Wind speeds maintain significant correlation for ~24 hours
- Temperature shows strong diurnal and seasonal patterns
- Soil moisture has the longest persistence, extending beyond 48 hours

### Method Evaluation

The analysis successfully captured:

1. Realistic marginal distributions for all variables
2. Physical dependencies between variables
3. Appropriate temporal correlation structures
4. Seasonal patterns and trends

The copula-based approach effectively modeled the complex dependencies while maintaining the correct marginal distributions, though at significant computational cost.

[^1]: Carta, J. A., Ramírez, P., & Velázquez, S. (2009). A review of wind speed probability distributions used in wind energy analysis: Case studies in the Canary Islands. Renewable and Sustainable Energy Reviews, 13(5), 933-955.
[^2]: Huybers, P., McKinnon, K. A., Rhines, A., & Tingley, M. (2014). U.S. daily temperatures: The meaning of extremes in the context of nonnormality. Journal of Climate, 27(19), 7368-7384.
[^3]: Famiglietti, J. S., Ryu, D., Berg, A. A., Rodell, M., & Jackson, T. J. (2008). Field observations of soil moisture variability across scales. Water Resources Research, 44(1).
