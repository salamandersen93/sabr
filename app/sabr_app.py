# /app/sabr_app.py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from modules.run_simulation_workflow_sqlite import SABRWorkflow
from modules.config import SIMULATION_PARAMS, INITIAL_STATE, KINETIC_PARAMS, REACTOR_PARAMS, SENSOR_PARAMS, FAULT_TEMPLATES

st.set_page_config(page_title="SABR: Synthetic Agentic Bioreactor", layout="wide")
st.title("SABR: Synthetic Agentic Bioreactor Simulation")

# --- Sidebar parameters ---
st.sidebar.header("Simulation Settings")
base_feed_rate = st.sidebar.slider("Base Feed Rate [g/L/h]", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
enable_anomaly = st.sidebar.checkbox("Enable Anomaly Detection", value=True)
enable_agent = st.sidebar.checkbox("Enable Agentic Analysis", value=True)
run_button = st.sidebar.button("Run Simulation")

# --- Run simulation ---
if run_button:
    with st.spinner("Running SABR simulation..."):
        workflow = SABRWorkflow(
            config_dict={
                "SIMULATION_PARAMS": SIMULATION_PARAMS,
                "INITIAL_STATE": INITIAL_STATE,
                "KINETIC_PARAMS": KINETIC_PARAMS,
                "REACTOR_PARAMS": REACTOR_PARAMS,
                "SENSOR_PARAMS": SENSOR_PARAMS,
                "FAULT_TEMPLATES": FAULT_TEMPLATES
            },
            enable_agent=enable_agent,
            enable_anomaly_detection=enable_anomaly,
            enable_agent_execution=enable_agent
        )

        results = workflow.run_with_monitoring(base_feed_rate=base_feed_rate)

    st.success(f"Simulation complete! Run ID: {results['run_id']}")

    # --- Agent explanation ---
    if enable_agent and results['agent_explain']:
        st.subheader("Agent Explanation / Root Cause Analysis")
        st.text(results['agent_explain'])

    # --- Telemetry visualization ---
    st.subheader("Telemetry Overview")
    df_true = results['true_history']
    df_obs = results['observed_history']

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    signals = ['X', 'S_glc', 'P', 'DO', 'pH']
    titles = ['Biomass [g/L]', 'Glucose [g/L]', 'Product Titer [g/L]', 
              'Dissolved Oxygen [%]', 'pH']

    for i, (sig, title) in enumerate(zip(signals, titles)):
        ax = axes[i]
        ax.plot(df_true['time'], df_true[sig], label='True', linewidth=2, alpha=0.8)
        ax.plot(df_obs['time'], df_obs[sig], label='Observed', linestyle='--', alpha=0.6)

        # Highlight anomalies
        if results['anomaly_scores']:
            sig_anomalies = [a for a in results['anomaly_scores'] 
                             if a.signal == sig and a.is_anomaly]
            anomaly_times = [a.time for a in sig_anomalies]
            if anomaly_times:
                anomaly_mask = df_obs['time'].isin(anomaly_times)
                ax.scatter(df_obs.loc[anomaly_mask, 'time'], 
                           df_obs.loc[anomaly_mask, sig],
                           color='red', marker='x', s=100,
                           label=f'Anomalies ({len(anomaly_times)})')

        ax.set_title(title)
        ax.set_xlabel('Time [h]')
        ax.grid(alpha=0.3)
        ax.legend()

    axes[-1].axis('off')
    st.pyplot(fig)

    # --- Run summary ---
    st.subheader("Run Summary")
    st.write({
        "Final Titer [g/L]": results['final_titer'],
        "Final Biomass [g/L]": results['final_biomass'],
        "Total Anomalies": results['num_anomalies']
    })

    st.success("Simulation complete ✅")