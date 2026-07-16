# FTSE Market Analysis - Unsupervised ML Project

## Live Application
* **Live App Link:** https://ftse-market-analysis-jlzhwszykpqzkkuzkzeqgt.streamlit.app/
* **Code Repository:** https://github.com
* **Project License:** [MIT](https://opensource.org)

---

## Project Overview
This project uses unsupervised machine learning to group FTSE stock market trading days and identify market anomalies. 

### Key Metrics
* **Selected Algorithm:** Hierarchical Clustering (Silhouette Score: 0.392)
* **K-Means Performance:** 4 clusters (Silhouette Score: 0.417)
* **Anomalies Detected:** 15 days flagged via Isolation Forest (5.02% of data)
* **DBSCAN Noise:** 58 days flagged as market outliers

---

## Project Structure
Use code with caution.ftse-market-analysis/
├── app/app.py                 # Streamlit web application dashboard
├── data/                      # Raw and preprocessed FTSE datasets
├── models/                    # Trained model files (.pkl, .npy)
├── src/                       # Python scripts for data prep, training, and evaluation
├── static/images/             # 7 evaluation and clustering plots
└── requirements.txt           # Python dependencies
---

## Setup & Deployment

### Local Run
```bash
git clone https://github.com
cd ftse-market-analysis
pip install -r requirements.txt
python src/data_prep.py
python src/train.py
streamlit run app/app.py
```

### Streamlit Cloud Deployment
1. Push this project repository to your GitHub account.
2. Log in to Streamlit Community Cloud.
3. Click New app, then select your repository, branch, and main file path (app/app.py).
4. Click Deploy. Streamlit Cloud will automatically read requirements.txt and build the application.

---

## Dashboard Logic

### Inputs
* Open, High, Low, Close prices, and Trading Volume.

### Outputs
* **K-Means Cluster:** Group assignment index (0-3).
* **DBSCAN / Anomaly Status:** Normal, Outlier, or Anomaly flag.
* **Suggestion:** 
  * Anomaly: "Unusual day detected - Be cautious!"
  * Cluster 1: "High activity - Consider aggressive trading"
  * Default: "Follow standard strategy"

---

## Future Goals
* Add technical metrics (RSI, MACD) and LSTM time-series forecasting.
* Set up live data pipeline streaming and automated model retraining.

**Author:** Amit Rajan
* GitHub: https://github.com/rajan7838
* Live Dashboard: https://ftse-market-analysis-jlzhwszykpqzkkuzkzeqgt.streamlit.app/