# FTSE Market Analysis: Unsupervised ML Project

[![Streamlit App](https://streamlit.io)](https://ftse-market-analysis-jlzhwszykpqzkkuzkzeqgt.streamlit.app/)
[![License: MIT](https://shields.io)](https://opensource.org)

An unsupervised machine learning project designed to cluster FTSE stock market trading days, identify structural patterns, and detect market anomalies using a live Streamlit dashboard.

🔗 **Live Dashboard:** [Launch Streamlit Application](https://ftse-market-analysis-jlzhwszykpqzkkuzkzeqgt.streamlit.app/)

---

## 📌 Project Overview
This repository uses clustering algorithms and anomaly detection models to parse historical FTSE trading data. By grouping days with similar price action and isolating volatility outliers, the system flags unique trading conditions automatically.

### 📊 Key Performance Metrics
* **Hierarchical Clustering:** 0.392 Silhouette Score
* **K-Means Clustering:** 4 distinct clusters (0.417 Silhouette Score)
* **Isolation Forest:** 15 extreme anomalies detected (5.02% of data)
* **DBSCAN:** 58 structural noise days flagged as market outliers

---

## 📂 Project Structure
```text
ftse-market-analysis/
├── app/
│   └── app.py              # Streamlit web application dashboard
├── data/                   # Raw and preprocessed FTSE datasets
├── models/                 # Trained model binaries (.pkl, .npy)
├── src/                    # Modular Python pipelines
│   ├── data_prep.py        # Data cleaning and scaling
│   └── train.py            # Model training and clustering execution
├── static/images/          # Saved evaluation and clustering plots
└── requirements.txt        # Python dependencies
```

---

## 🛠️ Setup & Local Deployment

### Local Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd ftse-market-analysis
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the data pipeline and train models:**
   ```bash
   python src/data_prep.py
   python src/train.py
   ```

4. **Launch the local Streamlit dashboard:**
   ```bash
   streamlit run app/app.py
   ```

### Streamlit Cloud Deployment
1. Fork or push this repository to your GitHub account.
2. Log in to [Streamlit Community Cloud](https://streamlit.io).
3. Click **New app**, then select your repository, branch, and main file path (`app/app.py`).
4. Click **Deploy**. Dependencies will auto-install from `requirements.txt`.

---

## 🧠 Dashboard Logic & Automation

### System Inputs
The models analyze 5 core market features: `Open`, `High`, `Low`, `Close` prices, and `Trading Volume`.

### Dynamic Output Strategy

| Condition | System Tag | Dashboard Action Suggestion |
| :--- | :--- | :--- |
| **Isolation Forest Flag** | `Anomaly` | ⚠️ "Unusual day detected - Be cautious!" |
| **K-Means Cluster 1** | `High Volatility` | ⚡ "High activity - Consider aggressive trading" |
| **Standard Baseline** | `Normal` | 🛡️ "Follow standard strategy" |

---

## 🚀 Future Goals
* [ ] Integrate technical analysis overlays (RSI, MACD, Bollinger Bands).
* [ ] Implement an LSTM time-series model for predictive forecasting.
* [ ] Build a live Kafka/API data pipeline stream with automated model retraining.

---

## 👤 Author
**Amit Rajan**
* GitHub: [@rajan7838](https://github.com/rajan7838)
* Project Link: [GitHub Repository](https://github.com)
