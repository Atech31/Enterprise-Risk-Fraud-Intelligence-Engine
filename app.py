import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split

# Page Configuration
st.set_page_config(
    page_title="ShieldRisk OS | Enterprise Risk Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom SaaS Dark Theme CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    
    .stApp {
        background-color: #0b0f19;
    }

    .brand-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 24px;
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        margin-bottom: 24px;
    }
    
    .brand-title {
        font-size: 20px;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.02em;
    }

    .badge-live {
        background-color: rgba(16, 185, 129, 0.1);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }

    [data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 20px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 13px !important;
        font-weight: 600 !important;
        color: #9ca3af !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 26px !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b !important;
    }

    .sidebar-footer {
        position: fixed;
        bottom: 20px;
        font-size: 14px;
        font-weight: 600;
        color: #94a3b8;
        padding: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Dataset Generator (Simulates Relational Financial Database)
@st.cache_data
def generate_transaction_data():
    np.random.seed(42)
    n = 1500
    
    data = pd.DataFrame({
        'Transaction_ID': [f'TXN{i:06d}' for i in range(1, n + 1)],
        'Account_ID': [f'ACC{np.random.randint(100, 300):04d}' for _ in range(n)],
        'Device_IP': [f'192.168.{np.random.randint(1, 10)}.{np.random.randint(1, 50)}' for _ in range(n)],
        'Amount_USD': np.random.exponential(scale=120, size=n).round(2) + 5,
        'Transaction_Velocity_1h': np.random.poisson(lam=2, size=n),
        'Distance_From_Last_Txn_KM': np.random.exponential(scale=40, size=n).round(1),
        'Is_High_Risk_Country': np.random.choice([0, 1], size=n, p=[0.92, 0.08]),
        'Hour_Of_Day': np.random.randint(0, 24, size=n)
    })
    
    fraud_score = (
        (data['Amount_USD'] > 500) * 0.35 +
        (data['Transaction_Velocity_1h'] > 5) * 0.30 +
        (data['Distance_From_Last_Txn_KM'] > 300) * 0.25 +
        (data['Is_High_Risk_Country'] == 1) * 0.30
    )
    data['Is_Fraud'] = (fraud_score > 0.45).astype(int)
    
    return data

df = generate_transaction_data()

# Model Training Pipeline
@st.cache_resource
def train_fraud_models(data):
    features = ['Amount_USD', 'Transaction_Velocity_1h', 'Distance_From_Last_Txn_KM', 'Is_High_Risk_Country', 'Hour_Of_Day']
    X = data[features]
    y = data['Is_Fraud']
    
    # Unsupervised Anomaly Detection
    iso = IsolationForest(contamination=0.08, random_state=42)
    data['Anomaly_Score'] = iso.fit_predict(X)
    data['Anomaly_Score'] = data['Anomaly_Score'].apply(lambda x: 1 if x == -1 else 0)
    
    # Supervised Classifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    return iso, rf, features

iso_model, rf_model, feature_cols = train_fraud_models(df)

# Sidebar Design
st.sidebar.markdown("### 🛡️ **ShieldRisk OS**")
st.sidebar.caption("v2.4 Pro | Financial Fraud Operations")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION", 
    [
        "📊 Live Risk Operations", 
        "🕸️ Fraud Ring Graph", 
        "⚙️ Loss & Threshold Engine"
    ]
)

st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-footer">👤 Platform Engineer: <b>Abhishek Ahire</b></div>', unsafe_allow_html=True)

# Shared Dark Theme Chart Layout
chart_layout = dict(
    font=dict(family="Plus Jakarta Sans", size=12, color="#9ca3af"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=20, r=20, t=30, b=20)
)

# Header Bar
st.markdown("""
    <div class="brand-header">
        <div class="brand-title">ShieldRisk OS — Enterprise Fraud & Risk Command</div>
        <div class="badge-live">● SYSTEM ACTIVE</div>
    </div>
""", unsafe_allow_html=True)

# Page 1: Live Risk Operations
if "Live Risk Operations" in page:
    st.markdown("##### **Executive Risk Telemetry**")
    
    col1, col2, col3, col4 = st.columns(4)
    total_volume = df['Amount_USD'].sum()
    fraud_count = df['Is_Fraud'].sum()
    fraud_rate = (fraud_count / len(df)) * 100
    flagged_amount = df[df['Is_Fraud'] == 1]['Amount_USD'].sum()
    
    col1.metric("Processed Volume", f"${total_volume:,.2f}")
    col2.metric("Total Transactions", f"{len(df):,}")
    col3.metric("Flagged Fraud Rate", f"{fraud_rate:.2f}%")
    col4.metric("At-Risk Capital", f"${flagged_amount:,.2f}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Transaction Amount Distribution**")
        fig_amt = px.histogram(
            df, x="Amount_USD", color="Is_Fraud", barmode="overlay",
            color_discrete_map={0: '#10b981', 1: '#ef4444'},
            labels={'Is_Fraud': 'Status'}
        )
        fig_amt.update_layout(**chart_layout)
        fig_amt.update_xaxes(showgrid=False, gridcolor='#1f2937')
        fig_amt.update_yaxes(showgrid=True, gridcolor='#1f2937')
        st.plotly_chart(fig_amt, use_container_width=True)
        
    with col_r:
        st.markdown("**Velocity vs Geographic Distance Deviation**")
        fig_scat = px.scatter(
            df, x="Distance_From_Last_Txn_KM", y="Transaction_Velocity_1h", color="Is_Fraud",
            size="Amount_USD", color_discrete_map={0: '#3b82f6', 1: '#ef4444'},
            labels={'Is_Fraud': 'Status'}
        )
        fig_scat.update_layout(**chart_layout)
        fig_scat.update_xaxes(showgrid=False)
        fig_scat.update_yaxes(showgrid=True, gridcolor='#1f2937')
        st.plotly_chart(fig_scat, use_container_width=True)

# Page 2: Fraud Ring Graph
elif "Fraud Ring Graph" in page:
    st.markdown("##### **Entity Relationship Network (Fraud Syndicate Detection)**")
    st.caption("Maps relationships between flagged accounts and shared device IPs to expose coordinated fraud rings.")
    st.markdown("<br>", unsafe_allow_html=True)
    
    fraud_subset = df[df['Is_Fraud'] == 1].head(35)
    
    G = nx.Graph()
    for _, row in fraud_subset.iterrows():
        G.add_node(row['Account_ID'], type='Account')
        G.add_node(row['Device_IP'], type='IP')
        G.add_edge(row['Account_ID'], row['Device_IP'])
        
    pos = nx.spring_layout(G, seed=42)
    
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, line=dict(width=1, color='#374151'),
        hoverinfo='none', mode='lines'
    )
    
    node_x, node_y, node_color, node_text = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        if G.nodes[node]['type'] == 'Account':
            node_color.append('#ef4444')
            node_text.append(f"Account: {node}")
        else:
            node_color.append('#f59e0b')
            node_text.append(f"Shared IP: {node}")
            
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', hoverinfo='text',
        text=node_text, textposition="top center",
        marker=dict(color=node_color, size=14, line=dict(width=1.5, color='#ffffff'))
    )
    
    fig_net = go.Figure(data=[edge_trace, node_trace])
    fig_net.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        **chart_layout
    )
    
    st.plotly_chart(fig_net, use_container_width=True)

# Page 3: Loss Engine
elif "Loss & Threshold Engine" in page:
    st.markdown("##### **Threshold Decisioning & Loss Optimization Simulator**")
    st.caption("Simulate business trade-offs between uncaught chargebacks and false-positive user friction.")
    st.markdown("<br>", unsafe_allow_html=True)
    
    probs = rf_model.predict_proba(df[feature_cols])[:, 1]
    df['Fraud_Probability'] = probs
    
    threshold = st.slider("Target Decision Cutoff Probability", 0.05, 0.95, 0.50, step=0.05)
    
    df['Predicted_Action'] = np.where(df['Fraud_Probability'] >= threshold, 'Block', 'Allow')
    
    false_positives = len(df[(df['Predicted_Action'] == 'Block') & (df['Is_Fraud'] == 0)])
    false_negatives = len(df[(df['Predicted_Action'] == 'Allow') & (df['Is_Fraud'] == 1)])
    true_positives = len(df[(df['Predicted_Action'] == 'Block') & (df['Is_Fraud'] == 1)])
    
    lost_revenue_fp = df[(df['Predicted_Action'] == 'Block') & (df['Is_Fraud'] == 0)]['Amount_USD'].sum() * 0.15
    uncaught_fraud_loss = df[(df['Predicted_Action'] == 'Allow') & (df['Is_Fraud'] == 1)]['Amount_USD'].sum()
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Blocked Fraud Count", f"{true_positives}")
    col_b.metric("Uncaught Fraud Loss", f"${uncaught_fraud_loss:,.2f}")
    col_c.metric("False Positive Cost", f"${lost_revenue_fp:,.2f}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("**Decision Outcome Impact**")
    fig_matrix = px.bar(
        x=["Blocked Fraud (True Pos)", "False Positive Blocks", "Uncaught Fraud Passes"],
        y=[true_positives, false_positives, false_negatives],
        color=["Blocked Fraud (True Pos)", "False Positive Blocks", "Uncaught Fraud Passes"],
        color_discrete_map={
            "Blocked Fraud (True Pos)": "#10b981",
            "False Positive Blocks": "#f59e0b",
            "Uncaught Fraud Passes": "#ef4444"
        }
    )
    fig_matrix.update_layout(**chart_layout, showlegend=False)
    fig_matrix.update_xaxes(showgrid=False)
    fig_matrix.update_yaxes(showgrid=True, gridcolor='#1f2937')
    st.plotly_chart(fig_matrix, use_container_width=True)
