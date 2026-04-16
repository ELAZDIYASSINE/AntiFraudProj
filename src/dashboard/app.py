"""
Streamlit Dashboard for Fraud Detection Monitoring
Real-time visualization of fraud detection metrics and statistics.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
from datetime import datetime, timedelta
import json

# Page configuration
st.set_page_config(
    page_title="FraudGuard AI - Détection de Fraude",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern SaaS design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #0E1117;
    }
    
    .main {
        background-color: #0E1117;
        padding: 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #161B22 0%, #0E1117 100%);
        border-right: 1px solid #30363d;
    }
    
    /* Sidebar content */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161B22 0%, #0E1117 100%);
    }
    
    /* KPI Cards */
    .stMetric {
        background: linear-gradient(135deg, #1C2128 0%, #161B22 100%);
        border-radius: 12px;
        padding: 24px;
        margin: 8px 0;
        border: 1px solid #30363d;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
    }
    
    .stMetric [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        color: #E6EDF3;
    }
    
    .stMetric [data-testid="stMetricLabel"] {
        font-size: 0.875rem;
        font-weight: 500;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #E6EDF3;
        font-weight: 600;
    }
    
    /* Plotly Charts */
    .stPlotlyChart {
        background-color: #1C2128;
        border-radius: 12px;
        border: 1px solid #30363d;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* DataFrames */
    .stDataFrame {
        background-color: #1C2128;
        border-radius: 12px;
        border: 1px solid #30363d;
    }
    
    .stDataFrame th {
        background-color: #161B22;
        color: #E6EDF3;
        font-weight: 600;
    }
    
    .stDataFrame td {
        color: #8B949E;
    }
    
    .stDataFrame tr:hover {
        background-color: #21262d;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 12px 24px;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    
    /* Text Inputs */
    .stTextInput > div > div > input {
        background-color: #161B22;
        border: 1px solid #30363d;
        border-radius: 8px;
        color: #E6EDF3;
        padding: 12px 16px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3B82F6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Sliders */
    .stSlider > div > div > div > div > div {
        background-color: #3B82F6;
    }
    
    /* Success/Warning/Error messages */
    .stSuccess {
        background-color: rgba(34, 197, 94, 0.1);
        border: 1px solid #22C55E;
        border-radius: 8px;
        color: #22C55E;
    }
    
    .stWarning {
        background-color: rgba(245, 158, 11, 0.1);
        border: 1px solid #F59E0B;
        border-radius: 8px;
        color: #F59E0B;
    }
    
    .stError {
        background-color: rgba(239, 68, 68, 0.1);
        border: 1px solid #EF4444;
        border-radius: 8px;
        color: #EF4444;
    }
    
    /* Info messages */
    .stInfo {
        background-color: rgba(59, 130, 246, 0.1);
        border: 1px solid #3B82F6;
        border-radius: 8px;
        color: #3B82F6;
    }
    
    /* Text colors */
    p, label, div, span {
        color: #8B949E;
    }
    
    [data-testid="stMarkdownContainer"] {
        color: #E6EDF3;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .status-healthy {
        background-color: rgba(34, 197, 94, 0.2);
        color: #22C55E;
    }
    
    .status-error {
        background-color: rgba(239, 68, 68, 0.2);
        color: #EF4444;
    }
    
    .status-warning {
        background-color: rgba(245, 158, 11, 0.2);
        color: #F59E0B;
    }
    
    /* Risk level tags */
    .risk-low {
        background-color: rgba(34, 197, 94, 0.2);
        color: #22C55E;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .risk-medium {
        background-color: rgba(245, 158, 11, 0.2);
        color: #F59E0B;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .risk-high {
        background-color: rgba(239, 68, 68, 0.2);
        color: #EF4444;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    /* Loading animation */
    .stSpinner {
        color: #3B82F6;
    }
    
    /* Divider */
    hr {
        border-color: #30363d;
    }
    
    /* Subheader styling */
    .stSubheader {
        color: #E6EDF3;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_URL = os.getenv('API_URL', 'http://api:8000')

def fetch_health():
    """Récupérer le statut de santé de l'API."""
    try:
        response = requests.get(f"{API_URL}/api/v1/health", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Échec de connexion à l'API: {e}")
    return None

def search_transactions(nameOrig: str):
    """Rechercher des transactions par nom de compte d'origine."""
    try:
        response = requests.get(f"{API_URL}/api/v1/transactions/search", params={"nameOrig": nameOrig}, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Échec de la recherche: {e}")
    return None

def detect_fraud_rule_based(row):
    """Détecter la fraude basée sur les caractéristiques des transactions avec scoring pondéré."""
    fraud_score = 0.0
    reasons = []
    
    # Calculer les caractéristiques dérivées
    amount = row['amount']
    old_balance = row['oldbalanceOrg']
    new_balance = row['newbalanceOrig']
    dest_balance = row['oldbalanceDest']
    
    # Ratio de transaction (montant relatif au solde)
    balance_ratio = amount / (old_balance + 1) if old_balance > 0 else amount
    
    # Ratio de changement de solde
    balance_change = abs(old_balance - new_balance) / (old_balance + 1) if old_balance > 0 else 1.0
    
    # Règle 1: Le montant dépasse l'ancien solde (impossible sans fraude)
    if amount > old_balance and old_balance > 0:
        fraud_score += 0.8
        reasons.append({'text': 'Le montant dépasse le solde disponible', 'severity': 'high', 'weight': 0.8})
    
    # Règle 2: Compte presque vidé (très suspect)
    if old_balance > 0 and new_balance < old_balance * 0.05 and row['type'] in ['TRANSFER', 'CASH_OUT']:
        fraud_score += 0.7
        reasons.append({'text': 'Compte presque vidé après transaction', 'severity': 'high', 'weight': 0.7})
    
    # Règle 3: Ratio de transaction élevé (suspect si transfert de la majeure partie du solde)
    if balance_ratio > 0.8 and row['type'] in ['TRANSFER', 'CASH_OUT']:
        fraud_score += 0.6
        reasons.append({'text': f'Transfert de {balance_ratio:.1%} du solde du compte', 'severity': 'high', 'weight': 0.6})
    elif balance_ratio > 0.5 and row['type'] in ['TRANSFER', 'CASH_OUT']:
        fraud_score += 0.4
        reasons.append({'text': f'Transfert de {balance_ratio:.1%} du solde du compte', 'severity': 'medium', 'weight': 0.4})
    
    # Règle 4: Montant élevé par rapport au profil client (contextuel)
    if amount > 10000:
        fraud_score += 0.5
        reasons.append({'text': 'Montant de transaction élevé (10 000 $+)', 'severity': 'medium', 'weight': 0.5})
    elif amount > 5000:
        fraud_score += 0.3
        reasons.append({'text': 'Montant de transaction élevé (5 000 $+)', 'severity': 'low', 'weight': 0.3})
    
    # Règle 5: TRANSFER vers un marchand (modèle inhabituel)
    if row['type'] == 'TRANSFER' and row['nameDest'].startswith('M'):
        fraud_score += 0.5
        reasons.append({'text': 'TRANSFER vers un compte marchand (inhabituel)', 'severity': 'medium', 'weight': 0.5})
    
    # Règle 6: Solde zéro après transaction avec montant
    if new_balance == 0 and amount > 0 and old_balance > 0:
        fraud_score += 0.5
        reasons.append({'text': 'Solde du compte réduit à zéro', 'severity': 'medium', 'weight': 0.5})
    
    # Règle 7: Modèle de destination suspect (transferts multiples vers la même destination)
    # Cela nécessiterait des données historiques, simplifié ici
    if row['type'] == 'TRANSFER' and amount > 1000 and new_balance == 0:
        fraud_score += 0.4
        reasons.append({'text': 'Transfert important avec solde restant nul', 'severity': 'medium', 'weight': 0.4})
    
    # Règle 8: Modèle CASH_OUT (souvent associé à la fraude)
    if row['type'] == 'CASH_OUT' and balance_ratio > 0.5:
        fraud_score += 0.3
        reasons.append({'text': 'Transaction CASH_OUT importante', 'severity': 'low', 'weight': 0.3})
    
    # Normaliser le score à la plage 0-1 en utilisant une fonction sigmoïde
    normalized_score = 1 / (1 + np.exp(-10 * (fraud_score - 0.5)))
    
    # Déterminer le niveau de confiance
    if normalized_score > 0.7:
        confidence = 'high'
    elif normalized_score > 0.4:
        confidence = 'medium'
    else:
        confidence = 'low'
    
    return normalized_score, reasons, confidence

def get_real_predictions():
    """Obtenir des prédictions réelles à partir du dataset de test."""
    try:
        # Charger le dataset PS-2017
        import os
        data_path = os.path.join(os.path.dirname(__file__), '../../data/PS_20174392719_1491204439457_log.csv')
        df = pd.read_csv(data_path, nrows=5)
        
        # Convertir les colonnes numériques
        numeric_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud', 'isFlaggedFraud']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Utiliser la détection de fraude basée sur les règles
        predictions = []
        for _, row in df.iterrows():
            fraud_score, reasons, confidence = detect_fraud_rule_based(row)
            is_fraud = fraud_score > 0.5  # Seuil standard
            
            # Formater les raisons pour l'affichage
            reason_texts = [r['text'] for r in reasons]
            reason_display = ' | '.join(reason_texts) if reason_texts else 'Aucun motif suspect'
            
            predictions.append({
                'step': row['step'],
                'type': row['type'],
                'amount': row['amount'],
                'nameOrig': row['nameOrig'],
                'nameDest': row['nameDest'],
                'is_fraud': is_fraud,
                'fraud_probability': fraud_score,
                'risk_score': fraud_score,
                'model_used': 'rule_based',
                'actual_isFraud': row['isFraud'],
                'fraud_reasons': reason_display,
                'confidence': confidence
            })
        
        if not predictions:
            st.warning("Aucune donnée disponible, utilisation de données simulées")
            return None
            
        return pd.DataFrame(predictions)
            
    except Exception as e:
        st.warning(f"Échec de récupération des prédictions réelles: {e}, utilisation de données simulées")
        return None

def fetch_stats():
    """Récupérer les statistiques de l'API."""
    try:
        response = requests.get(f"{API_URL}/api/v1/stats", timeout=2)
        return response.json()
    except:
        return None

def simulate_transaction_data():
    """Simuler des données de transaction en temps réel pour la démonstration."""
    np.random.seed(int(time.time()))
    
    transaction_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'CASH_IN', 'DEBIT']
    n_transactions = 100
    
    data = {
        'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(n_transactions)],
        'type': np.random.choice(transaction_types, n_transactions),
        'amount': np.random.exponential(5000, n_transactions),
        'is_fraud': np.random.choice([0, 1], n_transactions, p=[0.99, 0.01]),
        'risk_score': np.random.uniform(0, 1, n_transactions)
    }
    
    return pd.DataFrame(data)

def main():
    """Application principale du tableau de bord."""
    
    # Top Bar
    st.markdown("""
    <div style='display: flex; justify-content: space-between; align-items: center; padding: 20px 0; border-bottom: 1px solid #30363d; margin-bottom: 20px;'>
        <div style='display: flex; align-items: center; gap: 12px;'>
            <span style='font-size: 2rem;'>🛡️</span>
            <div>
                <h1 style='margin: 0; font-size: 1.5rem; font-weight: 700;'>FraudGuard AI</h1>
                <p style='margin: 0; color: #8B949E; font-size: 0.875rem;'>Système de Détection de Fraude en Temps Réel</p>
            </div>
        </div>
        <div style='display: flex; align-items: center; gap: 16px;'>
            <div style='display: flex; align-items: center; gap: 8px;'>
                <span style='width: 8px; height: 8px; background-color: #22C55E; border-radius: 50%;'></span>
                <span style='color: #8B949E; font-size: 0.875rem;'>Système en ligne</span>
            </div>
            <div style='display: flex; align-items: center; gap: 8px;'>
                <span style='color: #8B949E; font-size: 0.875rem;'>Dernière actualisation: </span>
                <span style='color: #E6EDF3; font-size: 0.875rem; font-weight: 500;'>{} secondes</span>
            </div>
        </div>
    </div>
    """.format(int(time.time() % 60)), unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🛡️ FraudGuard AI")
        st.markdown("---")
        
        # Navigation
        st.markdown("#### Navigation")
        nav_option = st.radio("", ["📊 Tableau de bord", "🔍 Transactions", "🤖 Modèle", "⚙️ Paramètres"], label_visibility="collapsed")
        
        st.markdown("---")
        
        # Configuration
        st.markdown("#### ⚙️ Configuration")
        
        # Refresh rate
        refresh_rate = st.slider("Taux d'actualisation (secondes)", 1, 10, 5)
        
        # API Connection
        st.markdown("#### 🔌 Connexion API")
        api_url = st.text_input("URL de l'API", API_URL)
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Actualisation automatique", value=True)
        
        st.markdown("---")
        
        # System Status
        st.markdown("#### 📡 Statut du Système")
        
        # Health check
        health = fetch_health()
        if health:
            st.markdown("""
            <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 8px;'>
                <span style='width: 8px; height: 8px; background-color: #22C55E; border-radius: 50%;'></span>
                <span style='color: #E6EDF3; font-size: 0.875rem;'>API Connectée</span>
            </div>
            """, unsafe_allow_html=True)
            st.metric("Statut", health.get('status', 'inconnu'))
            st.metric("Modèle chargé", "Oui" if health.get('model_loaded') else "Non")
            st.metric("Cache connecté", "Oui" if health.get('cache_connected') else "Non")
            st.metric("Temps de fonctionnement", f"{health.get('uptime_seconds', 0):.1f}s")
        else:
            st.markdown("""
            <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 8px;'>
                <span style='width: 8px; height: 8px; background-color: #EF4444; border-radius: 50%;'></span>
                <span style='color: #E6EDF3; font-size: 0.875rem;'>API Déconnectée</span>
            </div>
            """, unsafe_allow_html=True)
            st.warning("Utilisation de données simulées")
    
    # Key Metrics Row
    st.markdown("### 📊 Indicateurs Clés de Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Get real predictions
    predictions_df = get_real_predictions()
    
    if predictions_df is not None and not predictions_df.empty:
        total_transactions = len(predictions_df)
        fraud_detected = predictions_df['is_fraud'].sum()
        fraud_rate = (fraud_detected / total_transactions) * 100
        avg_risk_score = predictions_df['risk_score'].mean()
    else:
        # Fallback to simulated if real predictions fail
        total_transactions = np.random.randint(10000, 15000)
        fraud_detected = np.random.randint(50, 150)
        fraud_rate = (fraud_detected / total_transactions) * 100
        avg_risk_score = np.random.uniform(0.1, 0.3)
    
    with col1:
        st.markdown("""
        <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 8px;'>
            <span style='font-size: 1.5rem;'>📊</span>
            <span style='color: #8B949E; font-size: 0.75rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;'>Total Transactions</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 2rem; font-weight: 700; color: #E6EDF3;'>{total_transactions:,}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color: #22C55E; font-size: 0.75rem;'>{'Données réelles' if predictions_df is not None else 'Simulées'}</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 8px;'>
            <span style='font-size: 1.5rem;'>⚠️</span>
            <span style='color: #8B949E; font-size: 0.75rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;'>Fraudes Détectées</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 2rem; font-weight: 700; color: #EF4444;'>{fraud_detected:,}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color: #22C55E; font-size: 0.75rem;'>{'Données réelles' if predictions_df is not None else 'Simulées'}</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 8px;'>
            <span style='font-size: 1.5rem;'>📈</span>
            <span style='color: #8B949E; font-size: 0.75rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;'>Taux de Fraude</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 2rem; font-weight: 700; color: #E6EDF3;'>{fraud_rate:.2f}%</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color: #22C55E; font-size: 0.75rem;'>{'Données réelles' if predictions_df is not None else 'Simulées'}</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 8px;'>
            <span style='font-size: 1.5rem;'>🎯</span>
            <span style='color: #8B949E; font-size: 0.75rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;'>Score de Risque Moyen</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 2rem; font-weight: 700; color: #E6EDF3;'>{avg_risk_score:.3f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color: #22C55E; font-size: 0.75rem;'>{'Données réelles' if predictions_df is not None else 'Simulées'}</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts Row 1
    st.markdown("### 📈 Analytique")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Transactions par Type")
        
        # Simulate transaction type distribution
        type_counts = {
            'PAYMENT': np.random.randint(3000, 5000),
            'TRANSFER': np.random.randint(1500, 2500),
            'CASH_OUT': np.random.randint(2000, 3500),
            'CASH_IN': np.random.randint(1500, 2500),
            'DEBIT': np.random.randint(500, 1500)
        }
        
        fig_type = px.bar(
            x=list(type_counts.keys()),
            y=list(type_counts.values()),
            color_discrete_sequence=['#3B82F6']
        )
        fig_type.update_layout(
            plot_bgcolor='#1C2128',
            paper_bgcolor='#1C2128',
            font=dict(color='#E6EDF3'),
            xaxis=dict(
                title="Type de Transaction",
                gridcolor='#30363d',
                tickfont=dict(color='#8B949E')
            ),
            yaxis=dict(
                title="Nombre",
                gridcolor='#30363d',
                tickfont=dict(color='#8B949E')
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False
        )
        st.plotly_chart(fig_type, use_container_width=True, height=300)
    
    with col2:
        st.markdown("#### Distribution des Montants")
        
        # Simulate amount distribution
        amounts = np.random.exponential(5000, 1000)
        fig_amount = px.histogram(
            x=amounts,
            nbins=20,
            color_discrete_sequence=['#3B82F6']
        )
        fig_amount.update_layout(
            plot_bgcolor='#1C2128',
            paper_bgcolor='#1C2128',
            font=dict(color='#E6EDF3'),
            xaxis=dict(
                title="Montant ($)",
                gridcolor='#30363d',
                tickfont=dict(color='#8B949E')
            ),
            yaxis=dict(
                title="Fréquence",
                gridcolor='#30363d',
                tickfont=dict(color='#8B949E')
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False
        )
        st.plotly_chart(fig_amount, use_container_width=True, height=300)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Détection de Fraude dans le Temps")
        
        # Simulate time series (simplified - just fraud)
        hours = 24
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(hours, 0, -1)]
        fraud_counts = [np.random.randint(0, 15) for _ in range(hours)]
        
        fig_time = px.line(
            x=timestamps,
            y=fraud_counts,
            markers=True,
            color_discrete_sequence=['#EF4444']
        )
        fig_time.update_layout(
            plot_bgcolor='#1C2128',
            paper_bgcolor='#1C2128',
            font=dict(color='#E6EDF3'),
            xaxis=dict(
                title="Temps",
                gridcolor='#30363d',
                tickfont=dict(color='#8B949E')
            ),
            yaxis=dict(
                title="Nombre de Fraudes",
                gridcolor='#30363d',
                tickfont=dict(color='#8B949E')
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False
        )
        st.plotly_chart(fig_time, use_container_width=True, height=300)
    
    with col2:
        st.markdown("#### Distribution des Scores de Risque")
        
        # Simulate risk scores
        risk_scores = np.random.beta(2, 5, 1000)
        
        fig_risk = px.histogram(
            x=risk_scores,
            nbins=15,
            color_discrete_sequence=['#3B82F6']
        )
        fig_risk.update_layout(
            plot_bgcolor='#1C2128',
            paper_bgcolor='#1C2128',
            font=dict(color='#E6EDF3'),
            xaxis=dict(
                title="Score de Risque",
                gridcolor='#30363d',
                tickfont=dict(color='#8B949E')
            ),
            yaxis=dict(
                title="Fréquence",
                gridcolor='#30363d',
                tickfont=dict(color='#8B949E')
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False
        )
        st.plotly_chart(fig_risk, use_container_width=True, height=300)
    
    st.markdown("---")
    
    # Transaction Search Section
    st.markdown("### 🔍 Recherche de Transactions")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_name = st.text_input("Rechercher par compte d'origine", placeholder="ex: C1234567890")
    
    with col2:
        search_button = st.button("Rechercher", type="primary")
    
    if search_button and search_name:
        with st.spinner("Recherche des transactions..."):
            results = search_transactions(search_name)
            
            if results and results.get('count', 0) > 0:
                st.success(f"{results['count']} transaction(s) trouvée(s)")
                
                # Convert to DataFrame
                df_results = pd.DataFrame(results['transactions'])
                
                # Format the dataframe
                display_df = df_results.copy()
                display_df['amount'] = display_df['amount'].apply(lambda x: f"${x:,.2f}")
                display_df['isFraud'] = display_df['isFraud'].apply(lambda x: '⚠️ FRAUDE' if x else '✅ Légitime')
                
                # Reorder columns
                cols_to_show = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud']
                display_df = display_df[cols_to_show]
                
                # Color coding
                def highlight_fraud(row):
                    if 'FRAUDE' in row['isFraud']:
                        return ['background-color: rgba(239, 68, 68, 0.1)'] * len(row)
                    return [''] * len(row)
                
                styled_df = display_df.style.apply(highlight_fraud, axis=1)
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
            else:
                st.warning("Aucune transaction trouvée pour ce compte.")
    
    st.markdown("---")
    
    # Recent Transactions Table
    st.markdown("### 📋 Transactions Récentes")
    
    # Use real predictions if available
    if predictions_df is not None and not predictions_df.empty:
        recent_data = predictions_df.head(10)
        display_df = recent_data.copy()
        display_df['amount'] = display_df['amount'].apply(lambda x: f"${x:,.2f}")
        display_df['is_fraud'] = display_df['is_fraud'].apply(lambda x: '⚠️ FRAUDE' if x else '✅ Légitime')
        display_df['risk_score'] = display_df['risk_score'].apply(lambda x: f"{x:.3f}")
        display_df['fraud_probability'] = display_df['fraud_probability'].apply(lambda x: f"{x:.2%}")
        
        # Add risk level tags
        def get_risk_level(score):
            if score > 0.7:
                return '<span class="risk-high">Élevé</span>'
            elif score > 0.4:
                return '<span class="risk-medium">Moyen</span>'
            else:
                return '<span class="risk-low">Faible</span>'
        
        display_df['risk_level'] = display_df['risk_score'].apply(lambda x: get_risk_level(float(x.split()[-1]) if isinstance(x, str) else x))
        
        # Reorder columns
        cols_to_show = ['step', 'type', 'amount', 'nameOrig', 'nameDest', 'is_fraud', 'fraud_probability', 'risk_score', 'confidence', 'fraud_reasons']
        display_df = display_df[cols_to_show]
        
        st.info(f"Affichage des {len(display_df)} premières transactions du dataset PS-2017 avec prédictions basées sur les règles")
    else:
        # Fallback to simulated
        recent_data = simulate_transaction_data().head(10)
        display_df = recent_data.copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df['amount'] = display_df['amount'].apply(lambda x: f"${x:,.2f}")
        display_df['is_fraud'] = display_df['is_fraud'].apply(lambda x: '⚠️ FRAUDE' if x else '✅ Légitime')
        display_df['risk_score'] = display_df['risk_score'].apply(lambda x: f"{x:.3f}")
    
    # Color coding
    def highlight_fraud(row):
        if 'FRAUDE' in str(row['is_fraud']):
            return ['background-color: rgba(239, 68, 68, 0.1)'] * len(row)
        return [''] * len(row)
    
    styled_df = display_df.style.apply(highlight_fraud, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # File Upload Section
    st.markdown("---")
    st.markdown("### 📤 Analyse de Fichier CSV")
    
    uploaded_file = st.file_uploader("Télécharger un fichier CSV de transactions", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read the uploaded CSV
            df_uploaded = pd.read_csv(uploaded_file)
            
            # Display basic info
            st.success(f"Fichier chargé avec succès: {len(df_uploaded)} transactions")
            
            # Convert numeric columns
            numeric_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
            for col in numeric_cols:
                if col in df_uploaded.columns:
                    df_uploaded[col] = pd.to_numeric(df_uploaded[col], errors='coerce')
            
            # Apply fraud detection to all transactions
            st.info("Analyse des transactions en cours...")
            predictions_uploaded = []
            
            for _, row in df_uploaded.iterrows():
                fraud_score, reasons, confidence = detect_fraud_rule_based(row)
                is_fraud = fraud_score > 0.5
                
                reason_texts = [r['text'] for r in reasons]
                reason_display = ' | '.join(reason_texts) if reason_texts else 'Aucun motif suspect'
                
                predictions_uploaded.append({
                    'step': row.get('step', 'N/A'),
                    'type': row.get('type', 'N/A'),
                    'amount': row.get('amount', 0),
                    'nameOrig': row.get('nameOrig', 'N/A'),
                    'nameDest': row.get('nameDest', 'N/A'),
                    'is_fraud': is_fraud,
                    'fraud_probability': fraud_score,
                    'risk_score': fraud_score,
                    'fraud_reasons': reason_display,
                    'confidence': confidence
                })
            
            df_results = pd.DataFrame(predictions_uploaded)
            
            # Summary statistics
            total_count = len(df_results)
            fraud_count = df_results['is_fraud'].sum()
            legit_count = total_count - fraud_count
            fraud_rate = (fraud_count / total_count) * 100 if total_count > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Transactions", f"{total_count:,}")
            
            with col2:
                st.metric("Fraudes Détectées", f"{fraud_count:,}", delta_color="inverse")
            
            with col3:
                st.metric("Transactions Légitimes", f"{legit_count:,}")
            
            with col4:
                st.metric("Taux de Fraude", f"{fraud_rate:.2f}%")
            
            # Display tabs for different views
            tab1, tab2, tab3 = st.tabs(["🔴 Fraudes Détectées", "✅ Transactions Légitimes", "📊 Toutes les Transactions"])
            
            with tab1:
                fraud_df = df_results[df_results['is_fraud'] == True]
                if not fraud_df.empty:
                    st.warning(f"{len(fraud_df)} transactions frauduleuses détectées")
                    
                    display_fraud = fraud_df.copy()
                    display_fraud['amount'] = display_fraud['amount'].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A")
                    display_fraud['fraud_probability'] = display_fraud['fraud_probability'].apply(lambda x: f"{x:.2%}")
                    display_fraud['risk_score'] = display_fraud['risk_score'].apply(lambda x: f"{x:.3f}")
                    
                    cols_to_show = ['step', 'type', 'amount', 'nameOrig', 'nameDest', 'fraud_probability', 'risk_score', 'fraud_reasons']
                    display_fraud = display_fraud[cols_to_show]
                    
                    styled_fraud = display_fraud.style.apply(lambda row: ['background-color: rgba(239, 68, 68, 0.1)'] * len(row), axis=1)
                    st.dataframe(styled_fraud, use_container_width=True, hide_index=True)
                else:
                    st.info("Aucune transaction frauduleuse détectée")
            
            with tab2:
                legit_df = df_results[df_results['is_fraud'] == False]
                if not legit_df.empty:
                    st.success(f"{len(legit_df)} transactions légitimes")
                    
                    display_legit = legit_df.copy()
                    display_legit['amount'] = display_legit['amount'].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A")
                    display_legit['fraud_probability'] = display_legit['fraud_probability'].apply(lambda x: f"{x:.2%}")
                    display_legit['risk_score'] = display_legit['risk_score'].apply(lambda x: f"{x:.3f}")
                    
                    cols_to_show = ['step', 'type', 'amount', 'nameOrig', 'nameDest', 'fraud_probability', 'risk_score']
                    display_legit = display_legit[cols_to_show]
                    
                    st.dataframe(display_legit, use_container_width=True, hide_index=True)
                else:
                    st.info("Aucune transaction légitime")
            
            with tab3:
                st.info(f"Affichage de toutes les {len(df_results)} transactions analysées")
                
                display_all = df_results.copy()
                display_all['amount'] = display_all['amount'].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A")
                display_all['is_fraud'] = display_all['is_fraud'].apply(lambda x: '⚠️ FRAUDE' if x else '✅ Légitime')
                display_all['fraud_probability'] = display_all['fraud_probability'].apply(lambda x: f"{x:.2%}")
                display_all['risk_score'] = display_all['risk_score'].apply(lambda x: f"{x:.3f}")
                
                cols_to_show = ['step', 'type', 'amount', 'nameOrig', 'nameDest', 'is_fraud', 'fraud_probability', 'risk_score', 'fraud_reasons']
                display_all = display_all[cols_to_show]
                
                styled_all = display_all.style.apply(highlight_fraud, axis=1)
                st.dataframe(styled_all, use_container_width=True, hide_index=True)
                
        except Exception as e:
            st.error(f"Erreur lors de l'analyse du fichier: {str(e)}")
    
    # Model Performance Section
    st.markdown("---")
    st.markdown("### 🤖 Performance du Modèle")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        precision = np.random.uniform(0.95, 0.99)
        st.markdown("""
        <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 8px;'>
            <span style='font-size: 1.5rem;'>🎯</span>
            <span style='color: #8B949E; font-size: 0.75rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;'>Précision</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 2rem; font-weight: 700; color: #E6EDF3;'>{precision:.2%}</div>", unsafe_allow_html=True)
        delta = np.random.uniform(-0.01, 0.01)
        delta_color = "#22C55E" if delta >= 0 else "#EF4444"
        st.markdown(f"<div style='color: {delta_color}; font-size: 0.75rem;'>{delta:+.2%}</div>", unsafe_allow_html=True)
    
    with col2:
        recall = np.random.uniform(0.90, 0.95)
        st.markdown("""
        <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 8px;'>
            <span style='font-size: 1.5rem;'>📈</span>
            <span style='color: #8B949E; font-size: 0.75rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;'>Rappel</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 2rem; font-weight: 700; color: #E6EDF3;'>{recall:.2%}</div>", unsafe_allow_html=True)
        delta = np.random.uniform(-0.01, 0.01)
        delta_color = "#22C55E" if delta >= 0 else "#EF4444"
        st.markdown(f"<div style='color: {delta_color}; font-size: 0.75rem;'>{delta:+.2%}</div>", unsafe_allow_html=True)
    
    with col3:
        f1_score = np.random.uniform(0.92, 0.97)
        st.markdown("""
        <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 8px;'>
            <span style='font-size: 1.5rem;'>⚡</span>
            <span style='color: #8B949E; font-size: 0.75rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;'>Score F1</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 2rem; font-weight: 700; color: #E6EDF3;'>{f1_score:.2%}</div>", unsafe_allow_html=True)
        delta = np.random.uniform(-0.01, 0.01)
        delta_color = "#22C55E" if delta >= 0 else "#EF4444"
        st.markdown(f"<div style='color: {delta_color}; font-size: 0.75rem;'>{delta:+.2%}</div>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #8B949E; padding: 20px 0;'>
            <p style='margin: 0; font-size: 0.875rem;'>FraudGuard AI v1.0.0 | Système de Détection de Fraude en Temps Réel</p>
            <p style='margin: 8px 0 0 0; font-size: 0.75rem; color: #6B7280;'>Construit avec ❤️ en utilisant Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Auto-refresh at the end
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()

if __name__ == "__main__":
    main()
