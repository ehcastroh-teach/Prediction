# Predicting Housing Prices with Macroeconomic Indicators

Learn to build and compare regression models by predicting housing prices in Ames, Iowa, using both property-specific features and regional economic indicators. This course explores whether broader economic conditions (unemployment, manufacturing employment, consumer spending) improve upon traditional property-based prediction.

## Learning Objectives

By completing this course, you will be able to:

- Load, inspect, and clean datasets with missing values and temporal misalignments
- Aggregate time series data (monthly to quarterly) for analysis
- Identify feature importance through exploratory analysis and correlation
- Implement and evaluate multiple regression algorithms (Linear, Ridge, LASSO, Random Forest)
- Understand regularization and its role in preventing overfitting
- Build and interpret autoregressive time series models (AR, ARIMA, SARIMA)
- Compare cross-sectional vs. time-based prediction approaches
- Make informed choices about model selection based on problem requirements

## Data / File Dictionary

| File | Type | Purpose |
|------|------|---------|
| `01_data-cleaning.ipynb` | Jupyter Notebook | Load Ames dataset, handle missing values, aggregate macro data |
| `02_exploratory-analysis.ipynb` | Jupyter Notebook | Visualize distributions, compute correlations, detect patterns |
| `03_machine-learning-models.ipynb` | Jupyter Notebook | Train Linear, Ridge, LASSO, and Random Forest regressors |
| `04_time-series-methods.ipynb` | Jupyter Notebook | Build trend, AR, ARIMA, and SARIMA models |
| `materials/r-scripts/` | Directory | Original R implementations (reference) |

## Workflow Diagram

```
Raw Data (Ames Housing + Macro Indicators)
    ↓
[01_data-cleaning] → Inspect, clean, aggregate quarterly
    ↓
[02_exploratory-analysis] → Understand correlations and patterns
    ↓
    ├→ [03_machine-learning-models] → Cross-sectional prediction
    │   ├ Linear Regression
    │   ├ Ridge/LASSO
    │   └ Random Forest → Best: R² = 0.702
    │
    └→ [04_time-series-methods] → Time series prediction
        ├ Linear Trend
        ├ Random Walk
        ├ ARIMA
        └ SARIMA → Best: R² = 0.658

CONCLUSION: Cross-sectional approach outperforms time series
```

## Step-by-Step Walkthrough

### Notebook 1: Data Cleaning
**Why this matters**: Real-world data never arrives perfect. Before modeling, you must handle missing values, align time periods, and validate data integrity.

**What you'll do**:
1. Load housing sales and macroeconomic time series
2. Identify missing values and data type mismatches
3. Aggregate monthly economic data to quarterly periods
4. Merge quarterly macro indicators with housing sales
5. Validate the combined dataset before analysis

**Key insight**: Temporal aggregation (monthly → quarterly) was necessary because sales occur quarterly but economic data arrives monthly.

### Notebook 2: Exploratory Data Analysis
**Why this matters**: Before building models, understand what you're predicting and which features matter.

**What you'll do**:
1. Visualize the distribution of sale prices (right-skewed)
2. Calculate correlations between features and price
3. Identify property features that matter most (total square footage, condition)
4. Examine relationships with macroeconomic indicators
5. Detect time-based trends in pricing

**Key insight**: Total square footage (r=0.81) is the strongest single predictor. Macro indicators show moderate correlations (r=0.3-0.6), suggesting economic context matters but property features dominate.

### Notebook 3: Machine Learning Models (Cross-Sectional)
**Why this matters**: Property features + economic data provide rich information. Sophisticated models can find patterns linear methods miss.

**What you'll do**:
1. Train a baseline Linear Regression model
2. Apply Ridge and LASSO regularization to prevent overfitting
3. Build a Random Forest ensemble to capture non-linear relationships
4. Evaluate all models using R² and RMSE
5. Interpret feature importance from the best model

**Key insight**: Random Forest achieves R² = 0.702 (vs. 0.581 for linear regression). The improvement comes from capturing non-linear relationships and feature interactions—not from macro data specifically, but from better modeling the non-linearity that macro data alone cannot capture.

### Notebook 4: Time Series Methods
**Why this matters**: Some problems are best solved by forecasting trends, not modeling cross-sections. This notebook explores whether past prices predict future prices.

**What you'll do**:
1. Decompose prices into trend, seasonality, and noise
2. Fit simple models (linear trend, random walk)
3. Build autoregressive models that learn from recent price history
4. Implement ARIMA and SARIMA for more sophisticated patterns
5. Compare time-based approaches to cross-sectional methods

**Key insight**: Time series models achieve only R² ≈ 0.66 because they ignore property features. They answer a different question: "What will the market average be next quarter?" rather than "What is this specific home worth?"

## How to Run

### Prerequisites

- Python 3.7+
- Jupyter Notebook or JupyterLab
- Required packages: pandas, numpy, scikit-learn, statsmodels, matplotlib, seaborn

### Installation

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn jupyter
```

### Running the Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open and run notebooks in order:
# 1. 01_data-cleaning.ipynb
# 2. 02_exploratory-analysis.ipynb
# 3. 03_machine-learning-models.ipynb (or 04_time-series-methods.ipynb - independent)
# 4. 04_time-series-methods.ipynb (or 03_machine-learning-models.ipynb)
```

Each notebook is self-contained with example data. For real analysis, replace data loading sections with your own datasets.

## Key Concepts Glossary

- **Cross-sectional analysis**: Comparing many individuals at one point in time (many homes sold in Q2 2012)
- **Time series analysis**: Following one variable over time (Ames median price each quarter for 20 years)
- **Regularization**: Adding penalties to model coefficients to prevent overfitting (Ridge, LASSO)
- **Autoregressive**: A model where past values predict future values (AR, ARIMA)
- **Stationarity**: A time series with constant mean and variance over time (required for many time series methods)
- **Seasonality**: Predictable patterns that repeat (e.g., spring home sales higher than winter)
- **R² (Coefficient of Determination)**: Proportion of variance explained by a model (0 = no prediction, 1 = perfect)
- **RMSE (Root Mean Squared Error)**: Average prediction error in the same units as the target (dollars)
- **Macroeconomic indicators**: Regional economic metrics like unemployment rate, manufacturing employment, income levels

## Further Reading

- Introduction to Statistical Learning (ISLR) - Chapters on regression and ensemble methods
- Forecasting: Principles and Practice - Comprehensive time series resource
- Feature Engineering for Machine Learning - Understanding what makes good predictors
- Housing Economics and Market Analysis - Context for why housing prices matter

## Project Context

The Ames Housing Dataset is widely used in machine learning education. This project extends it by incorporating macroeconomic data - an exploration of whether buying behavior changes with regional economic conditions. The original motivation was to test whether economic indicators improve prediction accuracy relative to property features alone.

**Key Finding**: While macro indicators correlate with price (r ≈ 0.3-0.5), property-specific features are stronger predictors (r ≈ 0.6-0.8). A sophisticated non-linear model (Random Forest) using property features achieves R² = 0.702, slightly better than linear models that explicitly use macro data.

This reflects a broader principle: **what you're buying usually matters more than when you're buying it**, but understanding the economic context helps.

---

*This course teaches practical machine learning: data cleaning, exploratory analysis, model selection, and interpretation. By the end, you'll have built, compared, and understood multiple approaches to the same problem.*
---

## Contact

<div align="center">
  <img src="images/thumbnails/ehcastroh_teach_banner_flower.png" alt="ehcastroh" width="90" style="border-radius: 50%;" />

  <sub>ehcastroh</sub>

  <a href="https://github.com/ehcastroh">GitHub</a> · <a href="https://www.linkedin.com/in/ehcastroh/">LinkedIn</a>
</div>
