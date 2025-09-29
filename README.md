# synthetic_twin
Synthetic Twin is an end-to-end AI-driven bioprocess analytics platform designed to demonstrate synthetic bioreactor simulation, anomaly detection, and agentic reasoning. The platform combines three core capabilities:

Synthetic Bioreactor Twin 
- Generates plausible CHO fed-batch bioreactor runs (cell growth, substrate, product titer, host-cell protein, DO, pH).
- Includes realistic stochastic noise, drift, and offline assay sampling.
- Supports configurable feed schedules, setpoints, and reactor parameters.

Bioprocess Anomaly Detection Playground
- Detects deviations from expected process behavior (overfeeding, pH/DO excursions, sensor drift, contamination).
- Supports both classical methods (z-score, moving average, Isolation Forest) and ML-based detection (PyTorch/TensorFlow).
- Provides labeled synthetic anomalies for benchmarking and visualization.

Multi-Agent Bioprocess Copilot
- Monitoring Agent: Observes telemetry and raises alerts.
- Troubleshooting Agent: Interprets anomalies and proposes likely root causes.
- Reporting Agent: Generates natural language summaries and run reports.
- Optional agent-driven anomaly generation for complex, realistic scenarios.

The system is designed for personal exploration of AI in bioprocess analytics, leveraging synthetic data to avoid proprietary constraints.

Key Features

Interactive Dashboard:
- Real-time plots of bioreactor telemetry (cell density, substrate, DO, pH, product titer, HCP).
- Control panel for adjusting feed schedule, DO, pH setpoints, and injecting faults.
- Agent chat panel displaying recommendations, diagnostics, and explanations.
Synthetic Data Generation:
- Parameterized ODE-based simulation with configurable biological kinetics.
- Supports batch and streaming generation, stored in Delta Tables for scalable processing.
- Offline assays simulated daily (titer, HCP).
Fault Injection & Anomaly Labeling:
- Predefined fault archetypes: overfeeding, underfeeding, DO limitation, pH excursion, temperature spikes, sensor failures, contamination.
- Composite/agent-suggested anomalies for realism.
- Ground truth labels stored for evaluation and benchmarking.
Agentic Reasoning:
- Agents use telemetry + anomaly signals to propose interventions.
- LLM-driven summarization produces human-readable explanations.
- Modular agent architecture allows future expansion (Experiment Designer, QA reviewer, knowledge graph).

Architecture Overview
+--------------------+
| Synthetic Twin     |
| (Python/ODE model)|
+---------+----------+
          |
          v
+--------------------+
| Delta Tables / S3  |
| (Time series +     |
| anomalies)         |
+---------+----------+
          |
          v
+--------------------+
| ML / Anomaly       |
| Detection (sklearn/|
| PyTorch)           |
+---------+----------+
          |
          v
+--------------------+        +--------------------+
| Multi-Agent Copilot|<------>| LLM Summarization  |
| - Monitoring       |        | (LangChain / CrewAI)|
| - Troubleshooting  |        +--------------------+
| - Reporting        |
+---------+----------+
          |
          v
+--------------------+
| Frontend Dashboard |
| (React + TypeScript|
| + D3)              |
+--------------------+

Getting Started
Prerequisites

Databricks Community Edition account (Delta Tables)

Python 3.10+

Node.js + npm/yarn

Optional: OpenAI API key for LLM summarization

Installation (Local / Databricks)

Clone the repository:

git clone https://github.com/<username>/BioPilot.git
cd BioPilot


Python environment:

# For local dev
pip install -r requirements.txt


Frontend setup:

cd frontend
npm install
npm start


Start backend API (FastAPI):

cd backend
uvicorn main:app --reload


Load synthetic runs into Delta Tables (Databricks notebook included).

Demo Scenarios

Normal Run – Shows standard fed-batch behavior; agents summarize final titer and HCP.

Pump Omission / Overfeed Fault – Agent detects anomaly, proposes feed adjustment, summary updated.

Contamination Spike – Agent identifies root cause, demonstrates anomaly interpretation and downstream assay effect.

Each scenario is reproducible and comes with preloaded Delta table runs.

Folder Structure
/BioPilot
├─ backend/          # FastAPI API, data generators, ML models
├─ frontend/         # React + TypeScript + D3 dashboard
├─ notebooks/        # Databricks notebooks: simulation, anomaly detection, agents
├─ data/             # Sample Delta tables / synthetic runs
├─ agents/           # Agent modules and orchestration scripts
├─ requirements.txt
└─ README.md

Future Extensions

Multi-reactor simulation for parallel runs.

Experiment Designer agent (DoE proposal).

Knowledge graph linking parameters → anomalies → product quality.

LLM-enhanced scenario planning and counterfactual reasoning.

Exportable regulatory-style run reports.
