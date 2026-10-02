import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Football Analytics Hub", page_icon="⚽", layout="wide")
st.title("⚽ Football Analytics Dashboard")
st.markdown("Automated EDA, Form Tracking, and Match Dynamics for Predictive Modeling.")

# --- DATA PROCESSING (Cached for Performance) ---
@st.cache_data
def load_and_clean_data(files):
    # Load files and label seasons
    df1 = pd.read_csv(files[0])
    df2 = pd.read_csv(files[1])
    df1['Season'], df2['Season'] = 'Season 1', 'Season 2'
    df = pd.concat([df1, df2], ignore_index=True)
    
    # Standardize and Clean
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG']).copy()
    
    # Strip whitespace from categorical columns
    for col in ['HomeTeam', 'AwayTeam', 'FTR', 'HTR']:
        df[col] = df[col].astype(str).str.strip()

    # Enforce integer types
    for col in ['FTHG', 'FTAG', 'HTHG', 'HTAG']:
        df[col] = df[col].astype(int)

    # Filter out mathematically impossible rows
    df = df[(df['FTHG'] >= df['HTHG']) & (df['FTAG'] >= df['HTAG'])]
    
    # Feature Engineering (Macro & Decoupled Halves)
    df['TotalGoals'] = df['FTHG'] + df['FTAG']
    df['GoalDiff'] = df['FTHG'] - df['FTAG']
    df['SHHG'] = df['FTHG'] - df['HTHG']  # Second Half Home Goals
    df['SHAG'] = df['FTAG'] - df['HTAG']  # Second Half Away Goals
    df['HT_GoalDiff'] = df['HTHG'] - df['HTAG']
    df['SH_GoalDiff'] = df['SHHG'] - df['SHAG']
    df['Month'] = df['Date'].dt.month_name()
    
    return df

# --- SIDEBAR UI ---
st.sidebar.header("Data Upload")
uploaded_files = st.sidebar.file_uploader("Upload Two Season CSVs", type=['csv'], accept_multiple_files=True)

if len(uploaded_files) == 2:
    st.sidebar.success("Files loaded successfully.")
    df = load_and_clean_data(uploaded_files)
    
    # --- TOP LEVEL METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Matches", f"{len(df):,}")
    col2.metric("Total Goals Scored", f"{df['TotalGoals'].sum():,}")
    col3.metric("Home Win Rate", f"{(len(df[df['FTR'] == 'H']) / len(df) * 100):.1f}%")
    col4.metric("Away Win Rate", f"{(len(df[df['FTR'] == 'A']) / len(df) * 100):.1f}%")
    
    st.markdown("---")
    
    # --- DASHBOARD TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 Macro Trends", "⏱️ Match Dynamics (HT vs 2H)", "🛡️ Team Profiles"])
    
    with tab1:
        st.subheader("League Overview")
        col_a, col_b = st.columns(2)
        
        with col_a:
            # Match Outcomes
            fig_outcomes = px.pie(df, names='FTR', title="Full-Time Results Breakdown",
                                  color='FTR', color_discrete_map={'H':'#2ca02c', 'D':'#7f7f7f', 'A':'#d62728'}, hole=0.4)
            st.plotly_chart(fig_outcomes, use_container_width=True)
            
        with col_b:
            # Scoreline Frequency
            score_matrix = pd.crosstab(df['FTHG'], df['FTAG'])
            fig_score = px.imshow(score_matrix, text_auto=True, origin='lower',
                                  title="Exact Scoreline Frequency Matrix",
                                  labels=dict(x="Away Goals", y="Home Goals"), color_continuous_scale='YlGnBu')
            st.plotly_chart(fig_score, use_container_width=True)
            
    with tab2:
        st.subheader("First Half vs. Second Half Momentum")
        col_c, col_d = st.columns(2)
        
        with col_c:
            # 2D Density: HT vs 2nd Half Goals
            fig_density = px.density_heatmap(df, x='HTHG', y='SHHG', text_auto=True,
                                             title="Home Team: 1st Half vs 2nd Half Goals",
                                             labels={'HTHG': '1st Half Goals', 'SHHG': '2nd Half Goals'},
                                             color_continuous_scale='Blues')
            st.plotly_chart(fig_density, use_container_width=True)
            
        with col_d:
            # Momentum Shift (Goal Diff)
            df['HT_GD_Jitter'] = df['HT_GoalDiff'] + np.random.normal(0, 0.15, size=len(df))
            df['SH_GD_Jitter'] = df['SH_GoalDiff'] + np.random.normal(0, 0.15, size=len(df))
            fig_shift = px.scatter(df, x='HT_GD_Jitter', y='SH_GD_Jitter', opacity=0.4, color='FTR',
                                   title="Momentum Shift: HT Goal Diff vs 2H Goal Diff",
                                   labels={'HT_GD_Jitter': 'Half-Time Goal Diff (+ is Home Lead)', 
                                           'SH_GD_Jitter': 'Second-Half Goal Diff (+ is Home Lead)'},
                                   color_discrete_map={'H':'#2ca02c', 'D':'#7f7f7f', 'A':'#d62728'})
            fig_shift.add_hline(y=0, line_dash="dash", line_color="black")
            fig_shift.add_vline(x=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig_shift, use_container_width=True)

    with tab3:
        st.subheader("Team Specific Performance")
        # Top 10 Scoring Teams
        top_home = df.groupby('HomeTeam')['FTHG'].sum().nlargest(10).reset_index().sort_values('FTHG')
        fig_teams = px.bar(top_home, x='FTHG', y='HomeTeam', orientation='h',
                           title="Top 10 Most Lethal Home Teams (Total Goals)",
                           color='FTHG', color_continuous_scale='Reds')
        st.plotly_chart(fig_teams, use_container_width=True)

elif len(uploaded_files) > 0:
    st.sidebar.warning("Please upload exactly TWO CSV files to proceed.")
else:
    st.info("👈 Please upload your season CSV files in the sidebar to generate the dashboard.")