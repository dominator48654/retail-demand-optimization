# Retail Store Inventory — Demand Forecasting, Inventory Optimization & Dynamic Pricing

An end-to-end data science and machine learning project that combines **time-series demand forecasting, inventory optimization, and dynamic pricing** into a single retail analytics workflow.

The project uses a retail inventory dataset containing **73,000+ daily records** across multiple stores and products, with information related to sales, inventory, pricing, competition, weather, promotions, holidays, and seasonality.

The objective is to transform historical retail data into actionable decisions around:

- How much demand should be expected?
- Which forecasting model performs best?
- How much inventory should be maintained?
- When should a replenishment order be placed?
- What order quantity minimizes inventory costs?
- How should prices respond to demand and competitor pricing?
- What price maximizes estimated revenue?

---

## Project Overview

Retail businesses must simultaneously solve three interconnected problems:

### 1. Demand Forecasting

Accurately estimate future demand from historical sales and inventory patterns.

Four forecasting approaches are implemented and compared:

- **ARIMA**
- **Prophet**
- **LSTM**
- **GRU**

The models are evaluated using:

- Mean Absolute Error (**MAE**)
- Root Mean Squared Error (**RMSE**)
- Mean Absolute Percentage Error (**MAPE**)

The forecasting workflow includes stationarity analysis, seasonal decomposition, ACF/PACF analysis, chronological train-test splitting, model fitting, forecasting, and model comparison.

---

### 2. Inventory Optimization

Forecasting results are translated into inventory-management decisions.

The notebook calculates:

- **Safety Stock**
- **Reorder Point**
- **Economic Order Quantity (EOQ)**
- **Annual Ordering Cost**
- **Annual Holding Cost**
- **Total Annual Inventory Cost**

The objective is to balance:

> **Service level + stockout risk + inventory carrying cost + ordering cost**

The inventory model uses demand variability, supplier lead time, service-level assumptions, item cost, ordering cost, and holding-cost assumptions.

---

### 3. Dynamic Pricing

The project estimates the relationship between price and demand and uses it to identify a revenue-maximizing price.

The pricing analysis includes:

- Price-demand relationship
- Price elasticity of demand
- Promotion effectiveness
- Competitor price analysis
- Revenue-vs-price optimization
- Estimated optimal selling price
- Estimated revenue improvement

A log-log regression is used to estimate price elasticity:

```text
log(Demand) = α + β × log(Price) + ε
```

where **β represents the estimated price elasticity of demand**.

---

# Project Workflow

```text
                    ┌─────────────────────┐
                    │   Retail Dataset    │
                    │ 73,000+ Daily Rows  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Loading & EDA  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Cleaning &     │
                    │ Preprocessing       │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
       ┌─────────────┐  ┌──────────────┐  ┌──────────────┐
       │   Demand    │  │  Inventory   │  │   Dynamic    │
       │ Forecasting │  │ Optimization │  │   Pricing    │
       └──────┬──────┘  └──────┬───────┘  └──────┬───────┘
              │                │                │
              ▼                ▼                ▼
       ┌─────────────┐  ┌──────────────┐  ┌──────────────┐
       │ ARIMA       │  │ Safety Stock │  │ Elasticity   │
       │ Prophet     │  │ Reorder Point│  │ Promotions   │
       │ LSTM        │  │ EOQ          │  │ Competition  │
       │ GRU         │  │ Cost Analysis│  │ Price Opt.   │
       └──────┬──────┘  └──────┬───────┘  └──────┬───────┘
              │                │                │
              └────────────────┼────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Integrated Analytics│
                    │ Dashboard & Insights│
                    └─────────────────────┘
```

---

# Dataset

The project uses:

```text
retail_store_inventory.csv
```

The dataset contains **73,000+ daily records** covering multiple stores and products.

### Main Features

| Feature | Description |
|---|---|
| `Date` | Daily record date |
| `Store ID` | Unique store identifier |
| `Product ID` | Unique product identifier |
| `Category` | Product category |
| `Region` | Geographic region |
| `Inventory Level` | Stock available at the beginning of the day |
| `Units Sold` | Units sold during the day |
| `Units Ordered` | Units ordered |
| `Demand Forecast` | Existing demand forecast field |
| `Price` | Product selling price |
| `Discount` | Applied discount |
| `Competitor Pricing` | Competitor price |
| `Weather Condition` | Daily weather condition |
| `Holiday/Promotion` | Holiday or promotion indicator |
| `Seasonality` | Seasonal classification |

The notebook subsequently renames several columns to concise analysis-friendly names, including:

```text
Inventory Level      → Inventory
Units Sold           → Sales
Units Ordered        → Orders
Demand Forecast      → Demand
Weather Condition    → Weather
Holiday/Promotion   → Promotion
Competitor Pricing   → Competitor Price
```

---

# 1. Data Loading & Preprocessing

The notebook begins by loading the CSV dataset using Pandas.

```python
df = pd.read_csv('retail_store_inventory.csv')
```

Initial inspection includes:

- Dataset shape
- Data types
- Duplicate records
- Missing-value analysis
- Numerical-variable inspection
- Column renaming

The preprocessing stage creates cleaner variable names for downstream analysis and modelling.

---

# 2. Exploratory Data Analysis

The exploratory analysis examines the distribution of numerical variables and identifies potential outliers.

The notebook uses:

- Box plots
- Distribution analysis
- Time-series visualization
- Descriptive statistics

Plotly is used extensively for interactive visualizations.

---

# 3. Demand Forecasting

## Objective

Predict aggregate daily demand and compare classical statistical forecasting methods with deep-learning approaches.

The notebook aggregates demand across stores and products to create a single daily demand series:

```text
Store/Product Demand
        ↓
Daily Aggregation
        ↓
Aggregate Daily Demand
        ↓
Time-Series Forecasting
```

The forecasting dataset is split chronologically, with approximately **90% used for training and 10% reserved for testing**.

---

## 3.1 Stationarity Analysis

Before applying classical time-series models, the notebook evaluates stationarity using:

### Augmented Dickey-Fuller Test

The ADF test is used to assess whether the time series contains a unit root.

### KPSS Test

The KPSS test provides a complementary stationarity assessment because its null hypothesis differs from that of the ADF test.

Using both tests provides a more robust diagnostic.

---

## 3.2 Seasonal Decomposition

The demand series is decomposed into:

```text
Observed
   │
   ├── Trend
   ├── Seasonality
   └── Residual
```

A weekly seasonal period of **7 days** is used in the decomposition.

---

## 3.3 ACF & PACF

Autocorrelation and partial autocorrelation plots are generated to support ARIMA parameter selection.

These diagnostics help identify appropriate:

- AR order (`p`)
- Differencing order (`d`)
- MA order (`q`)

---

# 4. Forecasting Models

## ARIMA

A grid search is performed over candidate `(p, d, q)` combinations.

The candidate ranges are:

```text
p = 0 ... 3
d = 0 ... 2
q = 0 ... 3
```

The model with the lowest AIC is selected.

The selected model is then used to forecast the held-out test period.

Evaluation metrics:

- MAE
- RMSE
- MAPE

---

## Prophet

Prophet is fitted using:

- Weekly seasonality
- Yearly seasonality
- Additive seasonality

The model produces:

- Point forecasts
- Lower prediction bounds
- Upper prediction bounds
- Trend and seasonal components

MAPE is calculated with protection against division by zero.

---

## LSTM

A Long Short-Term Memory neural network is used to model sequential demand patterns.

### Configuration

The model uses:

- 30-day input sequences
- Min-Max scaling
- Two LSTM layers
- Dropout
- L2 regularization
- Dense layers
- Adam optimizer
- Huber loss
- Early stopping
- Learning-rate reduction

A key preprocessing consideration is that the scaler is fitted on the training data before being applied to the test data, reducing the risk of data leakage.

---

## GRU

A Gated Recurrent Unit network is implemented as a second deep-learning forecasting baseline.

The GRU architecture includes:

- Two GRU layers
- Dropout
- Dense layers
- Adam optimizer
- MSE loss
- Early stopping
- Learning-rate reduction

The GRU uses the same sequence-based forecasting setup so that its performance can be compared against the LSTM.

---

# 5. Model Evaluation

The four forecasting models are compared using a common evaluation framework.

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| ARIMA | Calculated in notebook | Calculated in notebook | Calculated in notebook |
| Prophet | Calculated in notebook | Calculated in notebook | Calculated in notebook |
| LSTM | Calculated in notebook | Calculated in notebook | Calculated in notebook |
| GRU | Calculated in notebook | Calculated in notebook | Calculated in notebook |

The notebook identifies the best-performing model based on the lowest MAPE.

The actual metric values should be taken from the notebook execution output rather than hard-coded into this README.

---

# 6. Inventory Optimization

## Objective

Translate demand information into practical inventory-control policies.

The analysis calculates:

### Safety Stock

Safety stock protects against demand variability during supplier lead time.

The implementation uses:

```text
Safety Stock = Z × σ × √Lead Time
```

where:

- `Z` = service-level z-score
- `σ` = demand standard deviation
- `Lead Time` = supplier lead time in days

---

## Reorder Point

The reorder point is calculated as:

```text
Reorder Point =
Demand During Lead Time + Safety Stock
```

This provides an inventory threshold at which a replenishment order should be initiated.

---

## Economic Order Quantity

EOQ determines an order quantity intended to balance ordering and holding costs.

The notebook uses:

```text
EOQ = √((2 × D × S) / H)
```

where:

- `D` = annual demand
- `S` = ordering cost per order
- `H` = annual holding cost per unit

---

## Inventory Cost Analysis

The notebook calculates:

```text
Annual Ordering Cost
Annual Holding Cost
Total Annual Inventory Cost
```

The total cost is:

```text
Total Annual Cost =
Annual Ordering Cost + Annual Holding Cost
```

---

## Inventory Assumptions

The notebook uses illustrative business assumptions for the optimization model, including:

| Parameter | Value |
|---|---:|
| Supplier Lead Time | 7 days |
| Holding Cost Rate | 25% annually |
| Stockout Cost | $50/unit |
| Item Cost | $25/unit |
| Target Service Level | 95% |
| Ordering Cost | $50/order |

These are **analytical assumptions**, not measured supplier contracts or actual business costs.

For production deployment, these parameters should be replaced with company-specific operational and financial data.

---

# 7. Dynamic Pricing

## Objective

Estimate how demand responds to price and identify a price that maximizes estimated revenue.

The pricing analysis considers:

- Product price
- Demand
- Discounts
- Competitor price
- Promotions

---

## Price Elasticity

Price elasticity is estimated using a log-log linear regression:

```text
log(Demand) = α + β × log(Price) + ε
```

The coefficient `β` represents estimated price elasticity.

A negative elasticity indicates that demand decreases as price increases.

The notebook also calculates the model's:

```text
R²
```

to quantify the explanatory power of the simple elasticity regression.

---

# 8. Promotion Analysis

The project compares average demand during:

```text
Promotion = 0
```

against:

```text
Promotion = 1
```

The resulting promotional lift is calculated as:

```text
Promotion Lift =
((Demand with Promotion - Demand without Promotion)
 / Demand without Promotion) × 100
```

This provides an initial estimate of the relationship between promotional activity and demand.

---

# 9. Revenue Optimization

The pricing model defines:

```text
Revenue = Price × Demand(Price)
```

A price range is generated from the observed price distribution, and estimated revenue is evaluated across that range.

The notebook identifies:

- Current average price
- Estimated current demand
- Current estimated revenue
- Revenue-maximizing price
- Demand at the optimal price
- Maximum estimated revenue
- Percentage price change
- Estimated revenue improvement

Competitor pricing is also incorporated into the simplified demand function.

---

# 10. Integrated Analytics Dashboard

The final notebook section combines the results from all three analytical challenges.

The integrated dashboard contains:

### Forecasting

- Model comparison
- Forecast accuracy

### Inventory

- Ordering cost
- Holding cost
- Total inventory cost

### Pricing

- Price-demand relationship
- Estimated elasticity
- Revenue-vs-price curve
- Optimal pricing point

The final workflow connects:

```text
Forecasting
     ↓
Demand Expectations
     ↓
Inventory Decisions
     ↓
Pricing Decisions
     ↓
Revenue & Cost Optimization
```

This creates a unified decision-support framework rather than treating forecasting, inventory, and pricing as independent analyses.

---

# Key Business Recommendations

The notebook produces a set of integrated recommendations:

1. Use the best-performing forecasting model for demand prediction.
2. Set inventory policies based on forecast variability and target service levels.
3. Use safety stock to protect against demand uncertainty.
4. Trigger replenishment around the calculated reorder point.
5. Use EOQ as a baseline for balancing ordering and holding costs.
6. Adjust pricing according to estimated demand elasticity.
7. Consider competitor pricing when evaluating price changes.
8. Analyze promotional timing against demand patterns.
9. Monitor forecasting performance continuously.
10. Retrain and recalibrate models as new demand data becomes available.

---

# Technology Stack

### Programming Language

- Python 3

### Data Processing

- Pandas
- NumPy

### Statistical Analysis

- Statsmodels
- SciPy

### Machine Learning

- Scikit-learn

### Deep Learning

- TensorFlow / Keras

### Forecasting

- ARIMA
- Prophet
- LSTM
- GRU

### Visualization

- Matplotlib
- Seaborn
- Plotly

### Development Environment

- Jupyter Notebook
- VS Code
- Anaconda / Python environment

---

# Project Structure

```text
forecasting-accuracy/
│
├── notebook.ipynb
│   └── Main analysis notebook
│
├── retail_store_inventory.csv
│   └── Retail inventory dataset
│
├── requirements.txt
│   └── Python dependencies
│
├── README.md
│   └── Project documentation
│
├── LICENSE
│   └── MIT License
│
└── .gitignore
    └── Git ignore configuration
```

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/dominator48654/forecasting-accuracy.git
cd forecasting-accuracy
```

## 2. Create a Virtual Environment

Using Python:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Or using Anaconda:

```bash
conda create -n retail-forecasting python=3.12
conda activate retail-forecasting
```

---

# Install Dependencies

Install the packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The notebook uses libraries including:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
statsmodels
prophet
tensorflow
scipy
plotly
```

---

# Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
notebook.ipynb
```

Alternatively, open the repository in VS Code and select a Python/Jupyter kernel with the required dependencies installed.

The notebook expects the dataset to be located in the same project directory:

```text
notebook.ipynb
retail_store_inventory.csv
```

The dataset is loaded using:

```python
pd.read_csv('retail_store_inventory.csv')
```

---

# Notebook Execution Order

For reproducible results, execute the notebook sequentially:

```text
1. Environment Setup
        ↓
2. Data Loading
        ↓
3. Data Cleaning
        ↓
4. Exploratory Data Analysis
        ↓
5. Demand Forecasting
        ├── Stationarity
        ├── ACF/PACF
        ├── ARIMA
        ├── Prophet
        ├── LSTM
        ├── GRU
        └── Model Comparison
        ↓
6. Inventory Optimization
        ├── Safety Stock
        ├── Reorder Point
        ├── EOQ
        └── Cost Analysis
        ↓
7. Dynamic Pricing
        ├── Elasticity
        ├── Promotion Analysis
        ├── Competitor Analysis
        └── Price Optimization
        ↓
8. Integrated Insights & Dashboard
```

---

# Visualization Note

The notebook uses Plotly for several interactive visualizations.

GitHub can render the notebook structure and static outputs, but interactive Plotly JavaScript visualizations may not behave the same way in GitHub's notebook viewer.

For the full interactive experience, open the notebook using:

- Jupyter Notebook
- JupyterLab
- VS Code
- Another compatible notebook environment

---

# Analytical Assumptions & Limitations

This project is designed as an end-to-end analytical demonstration and contains several simplifications.

### Aggregate Forecasting

The forecasting analysis aggregates demand across stores and products into a single daily time series.

A production system would generally benefit from hierarchical forecasting at levels such as:

```text
Total Business
     ↓
Region
     ↓
Store
     ↓
Category
     ↓
Product
```

### Inventory Assumptions

Lead time, item cost, ordering cost, holding cost, stockout cost, and service level are illustrative assumptions.

Real operational data should be used before applying the calculations to business decisions.

### Pricing Model

The dynamic-pricing component uses a simplified elasticity-based demand function.

Actual pricing systems could incorporate:

- Product-level elasticity
- Customer segmentation
- Competitor response
- Promotions
- Seasonality
- Cannibalization
- Cross-product effects
- Inventory constraints
- Margin requirements

### Forecasting Features

The forecasting models primarily operate on the aggregate demand time series.

Future versions could incorporate exogenous variables such as:

- Weather
- Promotions
- Holidays
- Competitor pricing
- Discounts

---

# Future Improvements

Potential extensions include:

### Forecasting

- Per-store forecasting
- Per-product forecasting
- Hierarchical forecasting
- ARIMAX
- SARIMA/SARIMAX
- XGBoost/LightGBM forecasting
- Hyperparameter optimization
- Rolling-window backtesting
- Probabilistic forecasting

### Inventory

- Multi-echelon inventory optimization
- Variable supplier lead times
- Service-level optimization
- Stockout simulation
- Safety-stock optimization
- Inventory policy simulation
- Forecast-uncertainty-aware replenishment

### Pricing

- Product-specific elasticity
- Segment-level pricing
- Competitor-aware pricing
- Promotion optimization
- Margin-constrained optimization
- Price experimentation
- Reinforcement-learning-based pricing

### Production Deployment

The analytical workflow could be converted into a production decision-support application with:

```text
Data Pipeline
      ↓
Forecasting Engine
      ↓
Inventory Optimization
      ↓
Pricing Engine
      ↓
Business Dashboard
      ↓
Automated Recommendations
```

---

# Results Interpretation

The notebook should be evaluated based on the outputs generated during execution.

For forecasting:

> Lower MAE, RMSE, and MAPE indicate better predictive performance.

For inventory:

> Lower total inventory cost must be considered alongside the target service level and stockout risk.

For pricing:

> The optimal price is the price producing the highest estimated revenue under the assumptions of the demand model.

The results should therefore be interpreted as **decision-support outputs rather than guaranteed business outcomes**.

---

# Reproducibility

To reproduce the analysis:

1. Clone the repository.
2. Install the required Python dependencies.
3. Ensure `retail_store_inventory.csv` is present.
4. Open `notebook.ipynb`.
5. Execute the notebook sequentially.
6. Review the forecasting comparison.
7. Review inventory optimization outputs.
8. Review dynamic pricing results.
9. Examine the integrated analytics dashboard.

---

# Project Highlights

```text
✓ 73,000+ retail records
✓ Time-series demand analysis
✓ Stationarity testing
✓ ADF + KPSS
✓ Seasonal decomposition
✓ ACF / PACF analysis
✓ ARIMA forecasting
✓ Prophet forecasting
✓ LSTM forecasting
✓ GRU forecasting
✓ MAE / RMSE / MAPE evaluation
✓ Model comparison
✓ Safety-stock calculation
✓ Reorder-point calculation
✓ EOQ optimization
✓ Inventory cost analysis
✓ Price elasticity estimation
✓ Promotion analysis
✓ Competitor price analysis
✓ Revenue optimization
✓ Integrated analytics dashboard
```

---

# Conclusion

This project demonstrates how multiple data-science techniques can be integrated into a unified retail decision-support workflow.

Instead of treating forecasting, inventory management, and pricing as separate problems, the project connects them:

```text
Historical Retail Data
        ↓
Demand Forecasting
        ↓
Inventory Planning
        ↓
Pricing Optimization
        ↓
Cost & Revenue Decisions
```

The resulting framework provides a foundation for building more sophisticated retail analytics systems capable of supporting demand planning, replenishment, and pricing decisions.

---

# License

This project is licensed under the **MIT License**.

See the `LICENSE` file for details.

---

# Author

**Harsh Verma**

Production and Industrial Engineering Student  
Birla Institute of Technology, Mesra

---

## Disclaimer

This project is intended for **educational, analytical, and portfolio purposes**.

The inventory and pricing calculations use simplified assumptions and should not be treated as direct operational recommendations without validation against real business data, costs, constraints, and historical performance.
