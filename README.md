# Retail Store Inventory — Demand Forecasting, Inventory Optimization & Dynamic Pricing

End-to-end data-science project on the **Retail Store Inventory** dataset (73,000+ daily records across
multiple stores and products, with sales, inventory, pricing, weather, promotion and holiday attributes).
The notebook addresses all three challenges posed by the dataset: time-series demand forecasting,
inventory optimization, and dynamic pricing.

## Challenges

1. **Time-Series Demand Forecasting** — predict daily demand and benchmark an **LSTM** against **ARIMA**,
   **Prophet** and a **GRU** on a common held-out window (MAE / RMSE / MAPE).
2. **Inventory Optimization** — derive **safety stock**, **reorder point** and **Economic Order Quantity
   (EOQ)** from demand variability and service-level targets, and break down annual holding vs. ordering cost.
3. **Dynamic Pricing** — estimate **price elasticity** (log-log regression), measure promotion lift, and
   optimize a revenue = price × demand(price) model to find a revenue-maximising price.

## Repository structure

```
.
├── notebook.ipynb        # Main analysis notebook (fully documented)
├── README.md
├── requirements.txt
└── .gitignore
```


## Notebook outline

1. Environment setup & imports
2. Data loading & initial inspection
3. Data cleaning & preprocessing
4. Exploratory data analysis (EDA)
5. Challenge 1 — Time-series demand forecasting (aggregate series → stationarity/decomposition →
   ACF/PACF → ARIMA → Prophet → LSTM → GRU → model comparison)
6. Challenge 2 — Inventory optimization
7. Challenge 3 — Dynamic pricing
8. Integrated insights & recommendations

## Getting started

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd <repo>

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add the dataset
#    place retail_store_inventory.csv in the project root

# 5. Launch Jupyter and run top to bottom
jupyter notebook notebook.ipynb
```

## Methodology highlights

- **Leakage-safe scaling:** the `MinMaxScaler` for the neural models is fitted on the training split only.
- **Robust metrics:** MAPE uses a zero-guard so zero-demand days don't distort the score; the LSTM trains
  with a Huber loss that is less sensitive to outliers.
- **Regularisation:** dropout, L2 weight decay, early stopping and learning-rate reduction guard against
  overfitting the deep models.
- **Two stationarity tests:** ADF and KPSS are used together (opposite null hypotheses) for a more
  confident verdict.

Concrete metric values, plots and the selected best model are shown inline in the notebook outputs.

## Notes

- Several figures use **Plotly**. GitHub's static notebook renderer does not run the embedded JavaScript,
  so interactive charts may appear blank on GitHub — view the notebook in Jupyter or via
  [nbviewer](https://nbviewer.org/) for full interactivity.
- The inventory cost, lead-time and service-level values are illustrative assumptions; replace them with
  real business figures for production use.

## Tech stack

Python 3.12 · pandas · NumPy · scikit-learn · statsmodels · Prophet · TensorFlow/Keras · Plotly ·
Matplotlib · seaborn · SciPy
