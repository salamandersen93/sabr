# Synthetic Agentic BioReactor (SABR)

> An interactive, AI-driven bioprocess simulation and analytics platform for synthetic bioreactor modeling, anomaly detection, and agent-assisted troubleshooting.

**Try it here:** https://sabr-app.streamlit.app/

First launch may take ~30 seconds while Streamlit Cloud spins up the application.

---

## Overview

**SABR** is a comprehensive platform that combines physics-based bioreactor simulation with intelligent fault detection and AI-powered troubleshooting. The system uses a simple yet effective modeling architecture for core bioreactor parameters and features a custom fault injection engine to create realistic, dynamic bioreactor runs. SABR was developed in Databricks Free Edition and deployed on Streamlit Cloud.

---

## Key Features

### **BioProcess End-to-End Simulation**
- Observe real-time bioreactor telemetry and assay outputs
- Collaborate with AI agents to diagnose faults, optimize feed strategies, and implement corrective actions
- Experience realistic CHO cell fed-batch dynamics with stochastic noise and sensor drift

### **Customized Run Mechanics**
- **Fault Diagnosis Challenges**: Detect hidden faults using limited telemetry and anomaly patterns
- **Scenario Difficulty Levels**: 
  - *Beginner* → Obvious, single-fault scenarios
  - *Intermediate* → Overlapping or subtle faults
  - *Expert* → Complex composite anomalies
- **Experiment Design Configuration**: Manipulate feed, DO, or pH setpoints to optimize runs
- **Interactive Feedback Loop**: AI agents provide hints and probabilistic reasoning without revealing solutions
- *(Future)* Dashboard with agent-assisted troubleshooting and gamified run diagnostics

### **Synthetic Data Generation**
- Fast, reproducible ODE-based simulation engine
- Randomized noise and drift models that mimic real sensor imperfections
- Configurable fault archetypes with labeled ground truth
- Data stored in Delta Tables for scalable advanced analytics

### **Agent Interaction**
- *(Current)* Troubleshooting and explainer agents interpret anomalies and suggest corrective actions
- *(Future)* Agents interact dynamically with user-selected assays and telemetry
- *(Future)* Multi-agent orchestration enabling reasoning across monitoring, troubleshooting, and reporting

### **Insightful Report Generation**
- AI-generated summaries with fault injection hints
- Comprehensive telemetry data and anomaly plots
- Full overview of run input parameters (kinetics, initial state, simulation parameters)
- Unique ID generation for run identification and targeted analytics in Delta Lake

---

## System Architecture

### Core Components

#### **1. Fed-Batch Game Engine** (Synthetic Bioreactor Twin)

The simulation heart of SABR, modeling realistic CHO cell fed-batch bioreactor dynamics.

**Simulated Parameters:**
- **Biomass (X)**: Cell density over time
- **Substrate (S)**: Nutrient concentration dynamics
- **Product/Titer (P)**: Product accumulation kinetics
- **Dissolved Oxygen (DO)**: Oxygen availability and transfer
- **pH**: Culture acidity/alkalinity
- *(Future)* Host Cell Protein (HCP) and Host Cell DNA (HCD)

**Key Features:**
- Incorporates stochastic noise and sensor drift for realistic telemetry
- User-defined assay panels for experimental design
- Outputs both continuous telemetry streams and discrete assay readouts

---

#### **2. Bioprocess Anomaly Detection**

Intelligent monitoring system that identifies deviations from expected bioreactor behavior.

**Detection Methods:**
- **Statistical Approaches**: Z-score analysis, moving average comparisons, control charts
- **Machine Learning**: Isolation Forest for multivariate anomaly detection
- *(Future)* Deep learning models using PyTorch (autoencoders, LSTMs)

**Supported Fault Archetypes:**
- Overfeeding (metabolic stress)
- Underfeeding (nutrient limitation)
- DO limitation (oxygen transfer issues)
- pH spikes (acid/base control failures)
- Temperature shifts (thermal excursions)
- Sensor failures (drift, bias, signal loss)
- Composite events (multiple simultaneous faults)
- *(Future)* Custom agent-introduced fault scenarios

---

####  **3. Multi-Agent Bioprocess Copilot**

AI-powered assistants that provide context-aware troubleshooting and optimization guidance.

**Current Capabilities:**
- **Troubleshooting/Explainer Agent**: Interprets anomalies, identifies likely root causes, and suggests diagnostic next steps
- Provides probabilistic reasoning and hints while maintaining user engagement

**Future Enhancements:**
- Agent-driven fault generation for complex, adaptive training scenarios
- Multi-agent orchestration for sophisticated reasoning across monitoring, troubleshooting, and reporting

---

## Module Structure

The platform is organized into modular Python components, orchestrated through a main Jupyter notebook:

### **`models.py`**
Contains bioreactor kinetic model definitions and deterministic functions for updating bioreactor state over time to generate telemetry data.

**Integration Role:** Core simulation engine that produces time-series data for all downstream modules.

---

### **`config.py`**
Centralized configuration hub containing:
- Run initialization parameters (starting conditions, default kinetics)
- Fault injection templates and archetypes
- Sensor specifications and characteristics

**Integration Role:** Single source of truth for simulation parameters, enabling reproducible runs and consistent fault injection.

---

### **`sensor_noise.py`**
Custom models for generating realistic sensor noise based on sensor type and characteristics.

**Integration Role:** Post-processes clean simulation data from `models.py` to add realistic measurement artifacts before anomaly detection.

---

### **`run_simulation_workflow.py`**
Core orchestration logic that executes the complete simulation pipeline.

**Integration Role:** Coordinates data flow between simulation, fault injection, noise generation, anomaly detection, and storage modules.

---

### **`anomaly_detection.py`**
Classical algorithms (z-score, moving average, Isolation Forest) that detect and flag anomalies in telemetry data.

**Integration Role:** Consumes noisy telemetry data and produces anomaly scores/flags that trigger agent investigation and reporting.

---

### **`agent_copilot.py`**
CrewAI-orchestrated Databricks LLaMA agents that troubleshoot and summarize bioreactor telemetry and anomaly data.

**Integration Role:** Interprets anomaly detection results and provides human-readable insights, recommendations, and interactive troubleshooting.

---

### **`data_lake.py`**
Logic for creating Delta Lake tables and writing simulation outputs for scalable advanced analytics.

**Integration Role:** Provides persistent storage layer with ACID transactions, enabling historical analysis and model training.

---

### **`reporting.py`**
Generates comprehensive PDF reports for each simulation scenario with download links (see examples in `/reports/`).

**Integration Role:** Synthesizes data from all modules (telemetry, anomalies, agent insights, fault labels) into cohesive documentation.

---

### **`parameter_sweep.py`**
Standalone script for running parameter sweeps—sets of simulations across multiple incrementing values for a given parameter.

**Integration Role:** Enables systematic exploration of parameter space for sensitivity analysis and optimization studies.

---

## End-to-End Integration Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER CONFIGURATION                           │
│           (config.py: initial conditions, kinetics)             │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                  SIMULATION EXECUTION                           │
│  run_simulation_workflow.py orchestrates:                       │
│    1. models.py → Generate clean bioreactor kinetics            │
│    2. (Optional) Fault injection from config.py templates       │
│    3. sensor_noise.py → Add realistic measurement artifacts     │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                      DATA STORAGE                               │
│    data_lake.py → Write telemetry to Delta Tables with         │
│                   unique run ID for traceability                │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                   ANOMALY DETECTION                             │
│  anomaly_detection.py → Analyze telemetry streams,             │
│                         flag deviations, calculate scores       │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                 AI-POWERED ANALYSIS                             │
│  agent_copilot.py → Interpret anomalies, provide context,      │
│                     suggest corrective actions                  │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                   REPORT GENERATION                             │
│  reporting.py → Compile comprehensive run documentation         │
│                 with AI summaries and visualizations            │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
                    USER REVIEW & ITERATION
```

---

## Future Extensions

- **Multi-reactor simulation**: Batch, perfusion, and scale-up scenarios
- **Reinforcement learning agents**: Optimized feed strategies through adaptive learning
- **Enhanced biochemistry**: Enzyme kinetics, thermodynamics, and metabolic byproducts
- **Knowledge graph**: Linking parameters → anomalies → product quality
- **Regulatory compliance**: Exportable regulatory-style run reports

---

## Getting Started

All modules are orchestrated through the main Jupyter notebook (MVP implementation). Simply configure your run parameters in `config.py` and execute the notebook cells to:

1. Generate synthetic bioreactor runs
2. Inject realistic fault scenarios
3. Detect anomalies in telemetry
4. Interact with AI troubleshooting agents
5. Generate comprehensive reports

---

## Example Reports

See the `/reports/` directory for example PDF reports demonstrating the full capabilities of SABR's reporting system.

---

## License

Copyright © 2025 Mike Andersen (salamandersen93)

All rights reserved. This project is currently a private portfolio piece.

---
