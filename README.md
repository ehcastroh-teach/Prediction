# Predicting Housing Prices with Macroeconomic Indicators

This repository teaches regression and time series forecasting through a single, concrete question: do regional economic conditions improve housing price prediction beyond what property features alone can explain? Using synthetic data modeled on the Ames, Iowa housing dataset, you will build and compare four regression models (Linear, Ridge, LASSO, Random Forest) and four time series models (Linear Trend, Random Walk, ARIMA, SARIMA), learning when and why to reach for each tool. The project also covers the full data science pipeline - from raw inspection and temporal aggregation through exploratory analysis, model evaluation, and interpretation - mirroring the decisions a practitioner makes on a real dataset.

## Learning Objectives

By completing this course, you will be able to:

- Load, inspect, and clean a structured dataset, identifying missing values, type mismatches, and outliers before touching a model
- Aggregate time series data from one frequency to another (monthly to quarterly) and justify the choice of aggregation function
- Merge datasets across time periods using a left join, and validate merge completeness programmatically
- Visualize price distributions, interpret right-skew, and decide when a log transform is warranted
- Compute and rank Pearson correlations, explain what a low correlation does and does not imply, and distinguish property-level from economy-level signal
- Train a Linear Regression baseline and explain why starting simple is methodologically sound
- Apply Ridge and LASSO regularization, explain the L2 vs. L1 penalty difference, and identify when each is preferable
- Build a Random Forest regressor, interpret feature importances, and articulate the bias-variance tradeoff in plain terms
- Decompose a price time series into trend, seasonality, and noise
- Implement AR, ARIMA, and SARIMA models, tune the (p, d, q) parameters, and test for stationarity
- Compare cross-sectional and time series approaches on the same prediction problem and articulate when each is the right tool

## Data / File Dictionary

| File | Type | Description |
|------|------|-------------|
| `01_data_cleaning.ipynb` | Jupyter Notebook | Inspect and clean a synthetic Ames-like dataset; aggregate monthly macro indicators to quarterly; merge by time period; validate the result |
| `02_exploratory_analysis.ipynb` | Jupyter Notebook | Visualize sale price distributions, rank feature correlations, examine macro-economic relationships, and detect time trends across the 2006-2010 period |
| `03_machine_learning_models.ipynb` | Jupyter Notebook | Train Linear, Ridge, LASSO, and Random Forest regressors; evaluate on R-squared and RMSE; compare interpretability vs. accuracy trade-offs |
| `04_time_series_methods.ipynb` | Jupyter Notebook | Build Linear Trend, Random Walk, AR, ARIMA, and SARIMA models on the quarterly price series; compare against cross-sectional results |
| `Price_Prediction_Using_R/clean_ames.R` | R Script | Original R implementation of data cleaning and missing-value handling |
| `Price_Prediction_Using_R/combine_macro_data.R` | R Script | Combines multiple FRED macro series into a single quarterly table |
| `Price_Prediction_Using_R/varPer_to_qtr.R` | R Script | Converts variable-period macro data to a uniform quarterly format |
| `Price_Prediction_Using_R/eda_house_prices.R` | R Script | Exploratory plots and correlation analysis in R |
| `Price_Prediction_Using_R/EDA.Rmd` | R Markdown | Rendered EDA report with narrative and figures |
| `Price_Prediction_Using_R/ames_ml_analysis.R` | R Script | Ridge, LASSO, and Random Forest regression in R using caret |
| `Price_Prediction_Using_R/time_dependence.R` | R Script | Tests for autocorrelation and time dependence in the price series |
| `Price_Prediction_Using_R/TS_Predicting_House_Prices.R` | R Script | ARIMA and SARIMA modeling in R using the forecast package |
| `Price_Prediction_Using_R/timeseries.Rmd` | R Markdown | Rendered time series report |
| `Price_Prediction_Using_R/stitch.R` | R Script | Utility to combine cleaned data sources into one analysis-ready file |
| `Price_Prediction_Using_R/utils.R` | R Script | Shared helper functions used across the R scripts |
| `requirements.txt` | Text | Python package versions needed to run the four notebooks |

## Workflow Diagram

```
Raw data (synthetic Ames housing + quarterly macro indicators)
    |
    v
[01_data_cleaning]
    Inspect shape, types, missing values
    Aggregate monthly macro -> quarterly average
    Left-merge housing sales to macro by (Year, Quarter)
    Validate: no nulls in key columns, no negative prices
    |
    v
[02_exploratory_analysis]
    Distribution of SalePrice (raw vs. log)
    Pearson correlations: property features vs. price
    Pearson correlations: macro indicators vs. price
    Quarterly price trend (2006-2010, spanning 2008 crash)
    |
    +-----------------------------+
    |                             |
    v                             v
[03_machine_learning_models]   [04_time_series_methods]
  Cross-sectional approach        Time series approach
  Input: property + macro         Input: quarterly price history
  Linear Regression baseline      Linear Trend model
  Ridge (L2 regularization)       Random Walk model
  LASSO (L1 regularization)       AR(p) autoregressive
  Random Forest (ensemble)        ARIMA(p,d,q)
  Best: Random Forest             SARIMA (seasonal extension)
  R-squared ~ 0.70                R-squared ~ 0.66

CONCLUSION: Cross-sectional approach outperforms time series
because it uses richer inputs (property features + macro data).
Time series answers a different question: market-wide trend forecasting.
```

## Step-by-Step Walkthrough

### Notebook 1 - Data Cleaning

**Why this comes first:** Modeling dirty data produces misleading results that are hard to diagnose. Cleaning and validation force you to understand the data before committing to any modeling strategy. The investment pays off throughout the rest of the project.

**Inspect before you fix.** The notebook opens by checking shape, dtypes, null counts, and descriptive statistics in that deliberate order. Shape gives the overall picture. Types reveal encoding errors - a price stored as a string will silently produce NaN when you try to compute a mean. Null counts determine which imputation strategy to apply. Statistics expose outliers. Each step sets up the next.

**Why average monthly values to quarterly - not sum?** Unemployment rate is a rate, not a count. Summing three months of unemployment rates would produce a meaningless number around 15-20 when the actual rate never exceeded 6.5%. Averaging preserves the original unit and meaning. Flow variables like total sales would be summed instead. The choice of aggregation function is not mechanical - it depends on what the number represents.

**Left-merge, not inner-merge.** Every housing sale row is preserved. Rows without a matching macro quarter get NaN macro values, which signals a gap in coverage rather than silently dropping observations. You check for these NaNs explicitly in the validation step.

**Programmatic validation over eyeballing.** The notebook closes with an assertion-based validator: no duplicate rows, no negative prices, year values in a plausible range, quarter values between 1 and 4. These checks run in under a second and catch the class of errors that are invisible in a 1,460-row table but immediately fatal in a downstream model.

### Notebook 2 - Exploratory Data Analysis

**Why spend a full notebook before modeling?** EDA generates hypotheses that guide model choice, feature engineering, and interpretation of results. Without EDA, you might fit a linear model to a non-linear relationship, overlook a transformation that would improve performance by 15%, or misinterpret a coefficient because you didn't know two features were correlated.

**Start with the target variable.** Housing prices are right-skewed - most homes cluster around a moderate price, with a long tail toward high-value properties. This is not noise; it reflects real estate markets worldwide. A log transform pulls the tail in and makes residuals more normally distributed, which improves the reliability of linear regression assumptions. Knowing this before fitting informs your modeling choices rather than discovering it after a poor residual plot.

**Correlation ranking shapes feature selection.** Total square footage is the dominant predictor among property features. Macro indicators (unemployment rate, manufacturing employment) show moderate positive or negative correlations with price. The key interpretive insight: a low correlation for macro indicators does not mean they are useless - it means they explain variance beyond what property size already accounts for. That is the exact hypothesis tested in Notebook 3.

**Time trends reveal the study period's context.** The 2006-2010 window covers the run-up to and aftermath of the 2008 financial crisis. If you model prices without acknowledging this trend, your model will be miscalibrated in ways that are hard to spot. Including year as a feature, or treating time explicitly as in Notebook 4, is the direct response to what EDA reveals here.

### Notebook 3 - Machine Learning Models (Cross-Sectional)

**Why start with Linear Regression?** It is not the most accurate model here, but it is the most informative starting point. The coefficients are directly interpretable: each additional square foot adds a specific dollar amount, holding all other features constant. The baseline R-squared tells you how much predictive signal exists in a purely linear relationship - everything above that in more complex models is attributable to non-linearity and interactions.

**Why regularize?** With multiple correlated features (total square footage and number of garage spaces are correlated; macro indicators are correlated with each other), ordinary least squares tends to assign large, unstable coefficients. Ridge (L2 penalty) shrinks all coefficients toward zero proportionally. LASSO (L1 penalty) drives some coefficients exactly to zero, effectively removing those features from the model. When you have a prior belief that only a subset of features matter, LASSO provides automatic feature selection without a separate selection step.

**Why Random Forest beats linear models here.** Housing price relationships are non-linear: an extra bedroom in a starter home adds more value than an extra bedroom in a mansion (diminishing returns). Garage spaces interact with neighborhood quality. Random Forest builds hundreds of decision trees on bootstrap samples, each using a random feature subset, and averages their predictions. This ensemble approach captures non-linear patterns and feature interactions that all three linear variants miss, at the cost of a model that cannot be summarized in a coefficient table.

**Model selection is a trade-off, not a competition.** The notebook concludes with a decision table: use Linear or LASSO when you need to explain a prediction to a homeowner or regulator; use Random Forest when maximizing accuracy; use LASSO when you have many potentially irrelevant features. Highest R-squared alone does not determine which model to deploy.

### Notebook 4 - Time Series Methods

**Why a separate notebook for time series?** Cross-sectional modeling (Notebook 3) asks: "Given this property and the current economic conditions, what is it worth?" Time series modeling asks: "Given that prices have followed this historical pattern, what will the average market price be next quarter?" These are different questions that require different data structures, different models, and different evaluation criteria.

**Decompose before modeling.** A price time series is the sum of trend (long-run direction), seasonality (predictable within-year patterns, e.g., spring sales peaks), and noise (random fluctuation). Modeling the composite series without separating these components means your model may chase seasonal patterns when the economic question is about trend, or vice versa.

**Why ARIMA improves on naive models.** A linear trend model assumes prices increase at a constant rate forever - implausible across market cycles. A random walk model assumes tomorrow equals today plus noise - reasonable for very short horizons but ignores exploitable patterns in recent history. An autoregressive (AR) model uses multiple past periods to predict the next one, capturing momentum. ARIMA adds differencing to remove trend non-stationarity and a moving average component to handle short-run residual autocorrelation. SARIMA extends ARIMA with seasonal lags.

**Why time series (R-squared ~ 0.66) loses to Random Forest (R-squared ~ 0.70).** Time series models use only price history as input. Random Forest uses property features and macro indicators - information that is simply unavailable to a time series model. The comparison is not a verdict that time series methods are inferior; it is a demonstration that input richness matters. A hybrid model using time series to forecast market trend combined with a cross-sectional model to price individual properties relative to that trend can outperform either alone.

## How to Run

### Prerequisites

- Python 3.8 or later
- pip

### Setup from a clean clone

```bash
git clone https://github.com/ehcastroh-teach/Prediction.git
cd Prediction

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the notebooks

```bash
jupyter notebook
```

Open and run notebooks in order. Each notebook is self-contained and regenerates its own synthetic dataset from a fixed random seed, so results are reproducible without downloading external data.

```
1. 01_data_cleaning.ipynb
2. 02_exploratory_analysis.ipynb
3. 03_machine_learning_models.ipynb
4. 04_time_series_methods.ipynb   (can also be run independently after notebook 2)
```

To verify notebooks execute without error from the command line:

```bash
jupyter nbconvert --to notebook --execute 01_data_cleaning.ipynb --output 01_data_cleaning_executed.ipynb
jupyter nbconvert --to notebook --execute 02_exploratory_analysis.ipynb --output 02_exploratory_analysis_executed.ipynb
jupyter nbconvert --to notebook --execute 03_machine_learning_models.ipynb --output 03_machine_learning_models_executed.ipynb
jupyter nbconvert --to notebook --execute 04_time_series_methods.ipynb --output 04_time_series_methods_executed.ipynb
```

### R scripts (optional reference)

The `Price_Prediction_Using_R/` directory contains the original R implementations used to develop this project. R 4.0+ with the `tidyverse`, `caret`, `forecast`, and `ggplot2` packages is required to run them. The Python notebooks are the primary learning path; the R scripts are reference material for learners familiar with R.

## Key Concepts Glossary

**Cross-sectional analysis** - Comparing many observations measured at the same point in time. In this project: predicting the price of each home sold in a given quarter using that home's features plus the macro conditions for that quarter.

**Time series analysis** - Tracking a single variable as it changes over time. Here: using the sequence of quarterly median prices to forecast the next quarter's median price.

**Temporal aggregation** - Converting data measured at one time frequency to a coarser frequency. Monthly unemployment rates become quarterly averages. The aggregation function (mean vs. sum) depends on whether the variable is a rate or a count.

**Left merge** - A table join that keeps all rows from the left table and attaches matching rows from the right table, filling unmatched rows with NaN. Used here to preserve every housing sale even when macro data is missing for that quarter.

**Regularization** - Adding a penalty on coefficient size to a regression objective. Prevents overfitting by discouraging the model from fitting noise in the training data. Ridge penalizes the sum of squared coefficients; LASSO penalizes the sum of absolute values.

**L1 vs. L2 penalty** - L1 (LASSO) drives some coefficients to exactly zero, performing implicit feature selection. L2 (Ridge) shrinks all coefficients smoothly toward zero but rarely reaches zero. Use LASSO when you believe many features are irrelevant; use Ridge when you want to retain all features but limit their magnitude.

**Random Forest** - An ensemble of decision trees, each trained on a random bootstrap sample of rows and a random subset of features. Predictions are averaged across all trees. The randomness reduces variance without increasing bias, and the ensemble captures non-linear relationships that linear models cannot.

**Stationarity** - A time series property: constant mean, constant variance, and no trend over time. Most time series forecasting methods (AR, ARIMA) require stationarity. You achieve it by differencing: subtracting each value from the previous one to remove trend.

**ARIMA(p, d, q)** - A time series model combining Autoregression (p past periods used as predictors), Integration (d rounds of differencing to achieve stationarity), and Moving Average (q past forecast errors used as predictors).

**SARIMA** - Seasonal ARIMA. Extends ARIMA with additional seasonal lag terms to capture repeating within-year patterns, such as spring price peaks in housing markets.

**R-squared (coefficient of determination)** - The proportion of variance in the target variable explained by the model. R-squared of 0 means the model does no better than predicting the mean every time; 1.0 means perfect prediction. Neither extreme is typical in practice.

**RMSE (Root Mean Squared Error)** - The square root of the average squared prediction error. Expressed in the same units as the target (dollars), making it directly interpretable. Squaring the errors penalizes large misses more heavily than small ones.

**Feature importance** - In a Random Forest, the average reduction in node impurity (or permutation-based measure) attributable to each feature across all trees. Provides a ranking of which inputs drive predictions most, partially compensating for the model's reduced interpretability relative to linear regression.

## Further Reading

- An Introduction to Statistical Learning - regression, regularization, and ensemble methods with worked examples
- Forecasting: Principles and Practice - comprehensive open-access reference for time series forecasting
- The Elements of Statistical Machine Learning - deeper theoretical treatment of the models covered in Notebook 3
- Feature Engineering for Machine Learning - practical guidance on constructing informative predictors
- Housing Economics and Urban Policy - context for why housing price modeling matters in practice

## Credits and Acknowledgements

The Ames Housing dataset was compiled by Dean De Cock and is widely used in machine learning education as a richer alternative to the Boston Housing dataset. The synthetic data in these notebooks is generated to match its statistical properties.

Macroeconomic indicators are modeled on data series published by the Federal Reserve Economic Data (FRED) system maintained by the Federal Reserve Bank of St. Louis.

The R implementations in `Price_Prediction_Using_R/` served as the original analysis from which the Python notebooks were developed.

---

## Contact

<div align="center">
  <img src="images/thumbnails/ehcastroh_teach_banner_flower.png" alt="ehcastroh" width="90" style="border-radius: 50%;" />

  <sub>ehcastroh</sub>

  <a href="https://github.com/ehcastroh">GitHub</a> · <a href="https://www.linkedin.com/in/ehcastroh/">LinkedIn</a>
</div>
