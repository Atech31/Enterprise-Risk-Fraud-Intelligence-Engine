#  Enterprise Risk & Fraud Intelligence Engine (ShieldRisk OS)

A production-ready Financial Fraud Analytics & Telemetry Engine built to evaluate real-time transaction velocities, identify geographical anomalies, uncover coordinated fraud rings using Graph Network Analysis, and simulate threshold-based loss trade-offs.

##  Live Demo
**Live Streamlit Application:** [https://enterprisefraud.streamlit.app/](https://enterprisefraud.streamlit.app/)

---

##  Architecture & Features

1. **Executive Risk Telemetry**
   - Live transaction volume monitoring, fraud rate calculation, and at-risk capital quantification.
   - Transaction distribution breakdown and velocity vs. distance anomaly detection.

2. **Graph Network Analysis (Fraud Ring Detection)**
   - Utilizes `NetworkX` graph structure to connect suspicious account IDs with shared device IPs.
   - Isolates coordinated fraud syndicates attempting multi-account chargeback abuse.

3. **Machine Learning Pipeline**
   - **Isolation Forest**: Unsupervised anomaly detection to flag unseen behavioral outliers.
   - **Random Forest Classifier**: Supervised classification engine calculating transaction risk probabilities.

4. **Threshold Loss Simulator**
   - Interactive slider allowing risk managers to set confidence cutoff thresholds.
   - Real-time trade-off modeling balancing **Uncaught Fraud Loss** against **False Positive Customer Friction Cost**.

---

##  Tech Stack
- **Language**: Python 3.10+
- **Frontend / UI**: Streamlit, Custom Dark SaaS CSS (`Plus Jakarta Sans`)
- **Data Engineering**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn (`IsolationForest`, `RandomForestClassifier`)
- **Graph & Visualization**: NetworkX, Plotly Express, Plotly Graph Objects

---

##  Local Setup & Run

1. Clone the repository:
   ```bash
   git clone [https://github.com/kunal-builds5/Enterprise-Risk-Fraud-Intelligence-Engine.git](https://github.com/kunal-builds5/Enterprise-Risk-Fraud-Intelligence-Engine.git)
   cd Enterprise-Risk-Fraud-Intelligence-Engine
