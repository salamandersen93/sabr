"""
dashboard.py

Streamlit-based frontend dashboard for BioPilot.
Provides interactive UI for running simulations, monitoring telemetry,
and interacting with the agent copilot.

Usage:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time

# Import BioPilot components (adjust paths as needed)
# from config import *
# from run_simulation_workflow import BioPilotWorkflow
# from agent_copilot import AgentObservation


# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="BioPilot - Bioreactor Training Simulator",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================

if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
    st.session_state.current_time = 0.0
    st.session_state.telemetry_data = []
    st.session_state.agent_actions = []
    st.session_state.anomalies = []
    st.session_state.run_id = None
    st.session_state.scenario = None
    st.session_state.score = 0
    st.session_state.assay_budget = 6

# ============================================
# SIDEBAR - SCENARIO SELECTION
# ============================================

st.sidebar.title("🧬 BioPilot Control Panel")
st.sidebar.markdown("---")

# Scenario selection
st.sidebar.subheader("Select Training Scenario")

scenarios = {
    "Tutorial: Perfect Run": {
        "description": "No faults, ideal conditions",
        "difficulty": "⭐ Easy",
        "faults": []
    },
    "Level 1: Feed Surge": {
        "description": "Single overfeed event",
        "difficulty": "⭐⭐ Medium",
        "faults": ["overfeed"]
    },
    "Level 2: Sensor Malfunction": {
        "description": "DO sensor freezes",
        "difficulty": "⭐⭐ Medium",
        "faults": ["sensor_freeze"]
    },
    "Level 3: Oxygen Crisis": {
        "description": "Critical DO drop",
        "difficulty": "⭐⭐⭐⭐ Hard",
        "faults": ["DO_drop"]
    },
    "Level 4: Cascade Failure": {
        "description": "Multiple simultaneous faults",
        "difficulty": "⭐⭐⭐⭐⭐ Expert",
        "faults": ["underfeed", "sensor_freeze", "pH_drift"]
    },
    "Expert: Contamination": {
        "description": "Detect bacterial contamination",
        "difficulty": "⭐⭐⭐⭐⭐ Expert",
        "faults": ["contamination"]
    }
}

selected_scenario = st.sidebar.selectbox(
    "Scenario",
    list(scenarios.keys())
)

st.sidebar.info(f"""
**{selected_scenario}**

{scenarios[selected_scenario]['description']}

**Difficulty**: {scenarios[selected_scenario]['difficulty']}
""")

# Simulation controls
st.sidebar.markdown("---")
st.sidebar.subheader("Simulation Controls")

if not st.session_state.simulation_running:
    if st.sidebar.button("🚀 Start Simulation", type="primary"):
        st.session_state.simulation_running = True
        st.session_state.scenario = selected_scenario
        st.session_state.current_time = 0.0
        st.session_state.telemetry_data = []
        st.session_state.agent_actions = []
        st.session_state.anomalies = []
        st.session_state.run_id = f"run_{int(time.time())}"
        st.rerun()
else:
    if st.sidebar.button("⏸️ Pause"):
        st.session_state.simulation_running = False
    
    if st.sidebar.button("🔄 Reset"):
        st.session_state.simulation_running = False
        st.session_state.current_time = 0.0
        st.session_state.telemetry_data = []
        st.rerun()

# Speed control
sim_speed = st.sidebar.slider(
    "Simulation Speed",
    min_value=1,
    max_value=10,
    value=5,
    help="Hours simulated per second"
)

# Display current status
st.sidebar.markdown("---")
st.sidebar.subheader("Current Status")
st.sidebar.metric("Run ID", st.session_state.run_id or "Not started")
st.sidebar.metric("Time Elapsed", f"{st.session_state.current_time:.1f} h")
st.sidebar.metric("Assays Remaining", f"{st.session_state.assay_budget}/6")
st.sidebar.metric("Score", f"{st.session_state.score:.0f} pts")

# ============================================
# MAIN DASHBOARD
# ============================================

st.title("🧬 BioPilot - CHO Cell Bioreactor Training Simulator")
st.markdown("""
Train on realistic bioprocess scenarios with AI-powered guidance. 
Detect anomalies, manage process parameters, and optimize your fed-batch run.
""")

# ============================================
# TELEMETRY PLOTS (Real-time)
# ============================================

st.header("📊 Live Telemetry")

# Create mock telemetry data if simulation running
if st.session_state.simulation_running and len(st.session_state.telemetry_data) < 200:
    # Simulate one timestep
    t = st.session_state.current_time
    
    # Mock data generation (replace with actual simulation)
    biomass = 0.1 + 0.03 * t + np.random.normal(0, 0.01)
    substrate = max(0.1, 20 - 0.15 * t + np.random.normal(0, 0.5))
    product = 0.01 * t ** 1.5 + np.random.normal(0, 0.01)
    DO = max(20, 100 - 0.4 * t + np.random.normal(0, 2))
    pH = 7.2 - 0.002 * t + np.random.normal(0, 0.01)
    
    st.session_state.telemetry_data.append({
        'time': t,
        'X': biomass,
        'S_glc': substrate,
        'P': product,
        'DO': DO,
        'pH': pH
    })
    
    st.session_state.current_time += 0.5  # dt
    
    # Simulate anomaly detection
    if DO < 30 and np.random.random() < 0.1:
        st.session_state.anomalies.append({
            'time': t,
            'signal': 'DO',
            'message': 'Critical DO level detected'
        })
    
    time.sleep(0.1 / sim_speed)  # Control speed
    st.rerun()

# Convert to DataFrame
if st.session_state.telemetry_data:
    df_telem = pd.DataFrame(st.session_state.telemetry_data)
    
    # Create multi-panel plot with Plotly
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Biomass [g/L]', 'Glucose [g/L]', 
                       'Product Titer [mg/mL]', 'Dissolved Oxygen [%]',
                       'pH', 'Feed Rate [g/L/h]'),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # Biomass
    fig.add_trace(
        go.Scatter(x=df_telem['time'], y=df_telem['X'], 
                  mode='lines', name='Biomass',
                  line=dict(color='green', width=2)),
        row=1, col=1
    )
    
    # Glucose
    fig.add_trace(
        go.Scatter(x=df_telem['time'], y=df_telem['S_glc'],
                  mode='lines', name='Glucose',
                  line=dict(color='blue', width=2)),
        row=1, col=2
    )
    
    # Product
    fig.add_trace(
        go.Scatter(x=df_telem['time'], y=df_telem['P'],
                  mode='lines', name='Product',
                  line=dict(color='purple', width=2)),
        row=2, col=1
    )
    
    # DO
    fig.add_trace(
        go.Scatter(x=df_telem['time'], y=df_telem['DO'],
                  mode='lines', name='DO',
                  line=dict(color='red', width=2)),
        row=2, col=2
    )
    
    # Add DO warning zone
    fig.add_hrect(y0=0, y1=30, 
                 line_width=0, fillcolor="red", opacity=0.1,
                 row=2, col=2)
    
    # pH
    fig.add_trace(
        go.Scatter(x=df_telem['time'], y=df_telem['pH'],
                  mode='lines', name='pH',
                  line=dict(color='orange', width=2)),
        row=3, col=1
    )
    
    # pH target zones
    fig.add_hrect(y0=6.8, y1=7.4,
                 line_width=0, fillcolor="green", opacity=0.1,
                 row=3, col=1)
    
    # Mark anomalies
    for anomaly in st.session_state.anomalies:
        fig.add_vline(x=anomaly['time'], line_dash="dash",
                     line_color="red", opacity=0.5)
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Real-Time Bioreactor Telemetry",
        title_font_size=20
    )
    
    fig.update_xaxes(title_text="Time [h]")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Biomass",
            f"{df_telem['X'].iloc[-1]:.2f} g/L",
            delta=f"{df_telem['X'].iloc[-1] - df_telem['X'].iloc[-2]:.3f}" if len(df_telem) > 1 else None
        )
    
    with col2:
        st.metric(
            "Glucose",
            f"{df_telem['S_glc'].iloc[-1]:.1f} g/L",
            delta=f"{df_telem['S_glc'].iloc[-1] - df_telem['S_glc'].iloc[-2]:.2f}" if len(df_telem) > 1 else None
        )
    
    with col3:
        st.metric(
            "Product Titer",
            f"{df_telem['P'].iloc[-1]:.2f} mg/mL",
            delta=f"{df_telem['P'].iloc[-1] - df_telem['P'].iloc[-2]:.3f}" if len(df_telem) > 1 else None
        )
    
    with col4:
        do_value = df_telem['DO'].iloc[-1]
        st.metric(
            "DO",
            f"{do_value:.1f} %",
            delta=f"{do_value - df_telem['DO'].iloc[-2]:.2f}" if len(df_telem) > 1 else None,
            delta_color="inverse"  # Low DO is bad
        )
    
    with col5:
        st.metric(
            "pH",
            f"{df_telem['pH'].iloc[-1]:.2f}",
            delta=f"{df_telem['pH'].iloc[-1] - df_telem['pH'].iloc[-2]:.3f}" if len(df_telem) > 1 else None
        )

else:
    st.info("👈 Start a simulation from the sidebar to begin monitoring")

# ============================================
# ANOMALY ALERTS
# ============================================

st.header("🚨 Anomaly Detection")

if st.session_state.anomalies:
    st.warning(f"**{len(st.session_state.anomalies)} anomalies detected**")
    
    # Show recent anomalies
    recent_anomalies = st.session_state.anomalies[-5:]
    for anomaly in reversed(recent_anomalies):
        with st.expander(f"⚠️ t={anomaly['time']:.1f}h - {anomaly['signal']} Alert"):
            st.write(anomaly['message'])
            st.write(f"**Signal**: {anomaly['signal']}")
            st.write(f"**Time**: {anomaly['time']:.1f} hours")
else:
    st.success("✅ No anomalies detected - process running normally")

# ============================================
# AGENT COPILOT CHAT
# ============================================

st.header("🤖 AI Copilot Assistant")

col_left, col_right = st.columns([2, 1])

with col_left:
    # Chat interface
    st.subheader("Chat with AI Copilot")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your BioPilot AI assistant. I can help you interpret telemetry, suggest interventions, and explain bioprocess concepts. How can I help?"}
        ]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask the copilot..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Mock response (replace with actual agent)
        if "DO" in prompt.upper() or "oxygen" in prompt.lower():
            response = "I notice the DO is dropping below 30%. I recommend increasing agitation to 120% immediately. Would you like me to make this adjustment?"
        elif "feed" in prompt.lower():
            response = "Current feed rate is 0.1 g/L/h. Substrate levels are stable at 15 g/L. No immediate adjustment needed."
        else:
            response = "Let me analyze the current process state and get back to you with recommendations."
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

with col_right:
    # Agent recommendations
    st.subheader("Recommended Actions")
    
    # Mock recommendations
    if df_telem is not None and not df_telem.empty:
        current_DO = df_telem['DO'].iloc[-1]
        
        if current_DO < 30:
            with st.container():
                st.error("🔴 **CRITICAL**: DO Below 30%")
                st.write("**Recommended Action**: Increase agitation")
                if st.button("✅ Execute: Increase Agitation +20%"):
                    st.session_state.agent_actions.append({
                        'time': st.session_state.current_time,
                        'action': 'Increase agitation',
                        'parameters': '+20%'
                    })
                    st.success("Action executed!")
                    st.rerun()
        
        substrate = df_telem['S_glc'].iloc[-1]
        if substrate < 5:
            with st.container():
                st.warning("🟡 **WARNING**: Low Substrate")
                st.write("**Recommended Action**: Increase feed rate")
                if st.button("✅ Execute: Increase Feed +10%"):
                    st.session_state.agent_actions.append({
                        'time': st.session_state.current_time,
                        'action': 'Increase feed',
                        'parameters': '+10%'
                    })
                    st.success("Action executed!")
                    st.rerun()
        
        # Assay request
        st.markdown("---")
        if st.button("🧪 Request Offline Assay", disabled=st.session_state.assay_budget <= 0):
            st.session_state.assay_budget -= 1
            st.success("Assay requested! Results in 4 hours.")
            st.rerun()

# ============================================
# ACTION HISTORY
# ============================================

st.header("📋 Action History")

if st.session_state.agent_actions:
    df_actions = pd.DataFrame(st.session_state.agent_actions)
    st.dataframe(df_actions, use_container_width=True)
else:
    st.info("No actions taken yet")

# ============================================
# KNOWLEDGE BASE (Learn More)
# ============================================

with st.expander("📚 Learn More: Bioreactor Fundamentals"):
    st.markdown("""
    ### Dissolved Oxygen (DO)
    - **Critical for**: Cell growth and metabolism
    - **Target range**: 30-80% saturation
    - **Low DO causes**: Reduced growth, cell death
    - **Control**: Adjust agitation, sparging rate
    
    ### pH Control
    - **Target range**: 6.8-7.4 for CHO cells
    - **Effect**: Impacts enzyme activity, growth rate
    - **Control**: Add acid/base, adjust CO2
    
    ### Feed Strategy
    - **Purpose**: Prevent substrate limitation/overfeeding
    - **Monitoring**: Glucose concentration
    - **Target**: 2-5 g/L during exponential phase
    
    ### Product Titer
    - **Growth-associated**: Produced during cell growth
    - **Non-growth-associated**: Basal production
    - **Target**: >8 mg/mL for commercial viability
    """)

with st.expander("🎯 Scenario Hints"):
    if st.session_state.scenario:
        scenario_hints = {
            "Tutorial: Perfect Run": [
                "Monitor all parameters - this is your baseline",
                "Notice how biomass, substrate, and product evolve",
                "Practice using the agent recommendations"
            ],
            "Level 1: Feed Surge": [
                "Watch for sudden substrate spike around t=20h",
                "High substrate can inhibit growth",
                "Consider reducing feed rate temporarily"
            ],
            "Level 2: Sensor Malfunction": [
                "DO sensor will freeze around t=50h",
                "Compare DO with biomass trends",
                "Use other signals to infer true oxygen status"
            ],
            "Level 3: Oxygen Crisis": [
                "Critical DO drop around t=60h",
                "Act immediately - cell death is irreversible",
                "Increase agitation to maximum"
            ],
            "Level 4: Cascade Failure": [
                "Multiple faults will occur",
                "Prioritize life-threatening issues first",
                "Use assays strategically to confirm interventions"
            ],
            "Expert: Contamination": [
                "Watch for unexpected death rate increase",
                "Biomass will plateau or decline",
                "Early detection is critical - use assays"
            ]
        }
        
        hints = scenario_hints.get(st.session_state.scenario, [])
        for i, hint in enumerate(hints, 1):
            st.markdown(f"{i}. {hint}")

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>BioPilot v1.0 | Developed for Bioprocess Training & Education</p>
    <p style='font-size: 0.8em; color: gray;'>
        🧬 Realistic CHO Kinetics | 🤖 AI-Powered Guidance | 📊 Delta Lake Persistence
    </p>
</div>
""", unsafe_allow_html=True)