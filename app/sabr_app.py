import sys
from pathlib import Path
import os

if "__file__" in globals():
    BASE_DIR = Path(__file__).resolve().parent
else:
    BASE_DIR = Path.cwd()
REPO_ROOT = BASE_DIR
for parent in BASE_DIR.parents:
    if (parent / "requirements.txt").exists() or (parent / ".git").exists():
        REPO_ROOT = parent
        break
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULES_DIR = REPO_ROOT / "app" / "modules"
if str(MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(MODULES_DIR))

# Now safe to import
from modules.run_simulation_workflow_sqlite import SABRWorkflow
from modules.config import (SIMULATION_PARAMS,INITIAL_STATE,KINETIC_PARAMS,
                            REACTOR_PARAMS,SENSOR_PARAMS,FAULT_TEMPLATES,)
from modules.reporting import BioreactorPDFReport

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from databricks.sdk import WorkspaceClient
import streamlit as st
from datetime import datetime

def get_secret(key):
    try:
        # Try Databricks secrets
        from pyspark.dbutils import DBUtils
        dbutils = DBUtils(spark)
        return dbutils.secrets.get(scope="sabr", key=key)
    except Exception as e:
        print('Unable to find databricks secrets with error:', e)
        print('falling back to streamlit secrets.')
        # Fallback to Streamlit secrets or environment variable
        try:
            import streamlit as st
            return st.secrets['databricks'][key]
        except Exception as e:
            print('unable to find streamlit secrets with error:', e)
            raise

host = get_secret("host")
token = get_secret("token")

client = WorkspaceClient(
    host=host,
    token=token)

st.set_page_config(page_title="SABR: Synthetic Agentic BioReactor", layout="wide")
st.title("SABR: Synthetic Agentic BioReactor Simulation")

# --- Sidebar parameters ---
st.sidebar.header("Simulation Settings")

# Fault selection
fault_options = list(FAULT_TEMPLATES.keys())
selected_fault = st.sidebar.selectbox("Fault Type", options=fault_options, index=0)

# Simulation parameters
st.sidebar.subheader("Simulation Parameters")
dt = st.sidebar.number_input("Time Step (dt) [h]", value=1.0, min_value=0.1, max_value=10.0, step=0.1)
total_time = st.sidebar.number_input("Total Time [h]", value=240, min_value=10, max_value=500, step=10)

# Initial state
st.sidebar.subheader("Initial State")
init_X = st.sidebar.number_input("Initial Biomass (X) [g/L]", value=0.1, min_value=0.01, max_value=5.0, step=0.01, format="%.3f")
init_S_glc = st.sidebar.number_input("Initial Glucose [g/L]", value=20.0, min_value=0.0, max_value=100.0, step=1.0)
init_pH = st.sidebar.number_input("Initial pH", value=7.20, min_value=6.0, max_value=8.0, step=0.01)

# Kinetic parameters (expandable)
with st.sidebar.expander("Kinetic Parameters"):
    mu_max = st.number_input("Max Growth Rate (mu_max) [1/h]", value=0.04, min_value=0.001, max_value=0.2, step=0.001, format="%.4f")
    Ks_glc = st.number_input("Glucose Half-Sat (Ks_glc) [g/L]", value=0.5, min_value=0.1, max_value=5.0, step=0.1)
    kd = st.number_input("Death Rate (kd) [1/h]", value=0.005, min_value=0.0, max_value=0.1, step=0.001, format="%.4f")

# Reactor parameters
with st.sidebar.expander("Reactor Parameters"):
    V0 = st.number_input("Initial Volume [L]", value=2.0, min_value=0.5, max_value=10.0, step=0.1)
    feed_start_h = st.number_input("Feed Start Time [h]", value=24.0, min_value=0.0, max_value=100.0, step=1.0)

base_feed_rate = st.sidebar.slider("Base Feed Rate [g/L/h]", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
enable_anomaly = st.sidebar.checkbox("Enable Anomaly Detection", value=True)
enable_agent = st.sidebar.checkbox("Enable Agentic Analysis", value=True)
run_button = st.sidebar.button("Run Simulation")

# --- Run simulation ---
if run_button:
    # Build config from UI inputs
    custom_config = {
        "SIMULATION_PARAMS": {**SIMULATION_PARAMS, "dt": dt, "total_time": int(total_time)},
        "INITIAL_STATE": {**INITIAL_STATE, "X": init_X, "S_glc": init_S_glc, "pH": init_pH},
        "KINETIC_PARAMS": {**KINETIC_PARAMS, "mu_max": mu_max, "Ks_glc": Ks_glc, "kd": kd},
        "REACTOR_PARAMS": {**REACTOR_PARAMS, "V0": V0, "feed_start_h": feed_start_h},
        "SENSOR_PARAMS": SENSOR_PARAMS,
        "FAULT_TEMPLATES": {selected_fault: FAULT_TEMPLATES[selected_fault]}
    }
    
    with st.spinner("Running SABR simulation..."):
        workflow = SABRWorkflow(
            config_dict=custom_config,
            enable_agent=enable_agent,
            enable_anomaly_detection=enable_anomaly,
            enable_agent_execution=enable_agent,
            host=host,
            token=token
        )
        results = workflow.run_with_monitoring(base_feed_rate=base_feed_rate)

    st.success(f"Simulation complete! Run ID: {results['run_id']}")

    # --- Agent explanation ---
    if enable_agent and results['agent_explain']:
        st.subheader("Agent Explanation / Root Cause Analysis")
        st.text(results['agent_explain'])

    # ---- Report generation ---
    reporter = BioreactorPDFReport()
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"/tmp/sabr_runs_{timestamp_str}"
    os.makedirs(output_folder, exist_ok=True)

    pdf_file = reporter.generate_summary_pdf(
        results=results,
        telemetry_df=pd.DataFrame(results['observed_history']),
        ai_summary=str(results['agent_explain']),
        faults=[FAULT_TEMPLATES[selected_fault]],
        param_config=custom_config,
        figures=[fig]
    )
    pdf_path = os.path.join(output_folder, f"{run_name}_report.pdf")
    shutil.move(pdf_file, pdf_path)
    print(f"PDF report saved to {pdf_path}")
    create_download_link(pdf_path)

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

    st.success("Simulation complete")