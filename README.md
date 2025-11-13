# Financial Anomaly Detection

This project focuses on **anomaly detection on financial transactions**.  
The goal is to detect unusual or suspicious transactions based on time, account, amount, merchant, transaction type and location.

Dataset source: [Financial Anomaly Data (Kaggle)](https://www.kaggle.com/datasets/devondev/financial-anomaly-data/data)

---

## Dataset

Main columns:

- **Timestamp** – Date and time of the transaction  
- **TransactionID** – Unique ID for each transaction  
- **AccountID** – ID of the account that made the transaction  
- **Amount** – Transaction amount  
- **Merchant** – Merchant / business where the transaction happened  
- **TransactionType** – Type of transaction (withdrawal, deposit, transfer, payment, etc.)  
- **Location** – Location of the transaction (city/country or store/portal code)

These features are used to analyze both **time-based** and **account/merchant-based** anomalies.

---

## Preprocessing / Feature Engineering

Current preprocessing steps:

- Convert `Timestamp` to a proper `datetime` type
- Create time-based features from `Timestamp`, such as:
  - `hour` (0–23)
  - `dayofweek` (0=Monday … 6=Sunday)
- Drop columns that are not very useful for modeling (e.g. `TransactionID` as a pure ID)
- Scale numerical features (e.g. `Amount`, `hour`, `dayofweek`)
- Encode categorical features using One-Hot Encoding:
  - `AccountID`
  - `Merchant`
  - `TransactionType`
  - `Location`

---

## Models

I plan to try different anomaly detection approaches:

- **PyTorch Autoencoder (Unsupervised)**
  - Train an autoencoder on transaction features
  - Use reconstruction error as an anomaly score per transaction

- **TensorFlow Models (Planned)**
  - Implement a similar dense autoencoder using TensorFlow / Keras
  - Optionally experiment with sequence models (LSTM/GRU) on per-account transaction histories

---

## Project Structure

```text
financial_anomly_detection/
├─ data/
│  └─ financial_anomaly_data.csv
├─ notebooks/
│  ├─ 01_eda.ipynb
│  ├─ 02_preprocessing.ipynb
│  ├─ 03_autoencoder_pytorch.ipynb
│  ├─ 04_gbdt.ipynb
│  └─ 05_tensorflow_models.ipynb
└─ README.md
