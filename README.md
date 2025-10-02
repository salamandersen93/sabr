# synthetic_twin
Project Overview

Synthetic Twin - Biopilot is an interactive, AI-driven bioprocess simulation and analytics platform designed to demonstrate synthetic bioreactor modeling, anomaly detection, and agent-assisted troubleshooting. BioPilot uses a simple modeling architecture for core bioreactor parameters and a custom fault injection engine to create plausible, dynamic bioreactor runs.

Fed-Batch Game Engine (Synthetic Bioreactor Twin)
- Simulates CHO fed-batch runs with core signals: Biomass (X), Substrate (S), Product/Titer (P), DO, and pH. (future: Host Cell Protein, Host Cell DNA)
- Incorporates stochastic noise and sensor drift for realism.
- Users can define assay panels, run experiments, and observe both telemetry and assay readouts.

Bioprocess Anomaly Detection
- Detects deviations in process behavior using classical analytics (z-score, moving average, Isolation Forest). (future: using ML - PyTorch)
- Fault archetypes include: overfeeding, underfeeding, DO limitation, pH spikes, temperature shifts, sensor failures, and composite events.
- (Future) Custom agent-introduced faults

Multi-Agent Bioprocess Copilot
- Troubleshooting/Explainer Agent: interprets anomalies, identifies likely root causes, and suggests next steps.
- (Future) agent-driven fault generation creates complex, realistic runs to challenge the user or other agents.

Key Features
1. BioProcess End-to-End Simulation
- Observe bioreactor telemetry and assay outputs.
- Work with agents to diagnose faults, optimize feed, or suggest corrective actions.

2. Customized Run Mechanics
- Fault Diagnosis Challenges: Hidden faults can be detected using limited telemetry and anomalies
- Scenario Levels: Beginner → obvious faults; Intermediate → overlapping or subtle faults; Expert → complex composite anomalies.
- Experiment Design Configuration: Users or agents can manipulate feed, DO, or pH setpoints to optimize runs.
- Feedback Loop: Agents provide hints and probabilistic reasoning without giving away solutions.
- (Future) dashboard with agent-assisted troubleshooting at run level (gamification of run troubleshooting)

3. Synthetic Data Generation
- Fast, reproducible ODE-based simulation.
- Randomized noise and drift mimic sensor imperfections.
- Configurable faults/anomalies with labeled ground truth.
- Data stored in Delta Tables for scalable analytics.

4. Agent Interaction
- (Future) agents interact dynamically with user-selected assays and telemetry.
- (Future) ulti-agent orchestration enables reasoning across monitoring, troubleshooting, and reporting.

Architecture Overview

Fed-Batch Game Engine --> Delta Tables/S3 (Synthetic Runs) --> ML Anomaly Detection (PyTorch/sklearn) --> Multi-Agent Copilot (troubleshooting, monitoring, reporting) --> LLM Summarization --> Frontend

Future Extensions
- Multi-reactor simulation (batch, perfusion, scale-up).
- Reinforcement learning agents for optimized feed strategies.
- Integration of enzyme kinetics, thermodynamics, or metabolic byproducts.
- Knowledge graph linking parameters → anomalies → product quality.
- Exportable regulatory-style run reports.

High Level Architecture:

<img width="330" height="602" alt="image" src="https://github.com/user-attachments/assets/8783171d-d276-44e1-9ffd-6142362f436c" />
