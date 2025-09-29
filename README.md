# synthetic_twin
Project Overview

Synthetic Twin - Biopilot is an interactive, AI-driven bioprocess simulation and analytics platform designed to demonstrate synthetic bioreactor modeling, anomaly detection, and agent-assisted troubleshooting within a gamified learning environment. BioPilot uses a “game engine” approach to create plausible, dynamic bioreactor runs.

Fed-Batch Game Engine (Synthetic Bioreactor Twin)
- Simulates CHO fed-batch runs with core signals: Biomass (X), Substrate (S), Product/Titer (P), DO, pH, and Host-Cell Protein (HCP).
- Incorporates stochastic noise, drift, and assay lag for realism.
- Users can define assay panels, run experiments, and observe both telemetry and assay readouts.

Bioprocess Anomaly Detection Playground
- Detects deviations in process behavior using classical analytics (z-score, moving average, Isolation Forest) and ML (PyTorch).
- Fault archetypes include: overfeeding, underfeeding, DO limitation, pH spikes, temperature shifts, sensor failures, and composite events.
- Labels can be algorithmically injected for benchmarking, or agent-generated for advanced scenarios.

Multi-Agent Bioprocess Copilot
- Monitoring Agent: observes process telemetry and flags anomalies.
- Troubleshooting Agent: interprets anomalies, identifies likely root causes, and suggests next steps.
- Reporting Agent: produces human-readable summaries and run reports.
- Optional agent-driven fault generation creates complex, realistic runs to challenge the user or other agents.

Key Features
1. Interactive Gameplay
- Select assay panels dynamically for each simulated run.
- Observe bioreactor telemetry and assay outputs in real time.
- Work with agents to diagnose faults, optimize feed, or suggest corrective actions.

2. Gamification Mechanics
- Fault Diagnosis Challenges: Hidden faults must be detected using limited telemetry or assays.
- Resource Management: Limited assay budget encourages strategic measurement decisions.
- Scenario Levels: Beginner → obvious faults; Intermediate → overlapping or subtle faults; Expert → complex composite anomalies.
- Experiment Designer Mode: Users or agents can manipulate feed, DO, or pH setpoints to optimize runs.
- Feedback Loop: Agents provide hints and probabilistic reasoning without giving away solutions.
- Achievements & Leaderboards: Track accuracy, efficiency, and optimal runs.

3. Synthetic Data Generation
- Fast, reproducible ODE-based “game engine” simulation.
- Randomized noise and drift mimic sensor imperfections.
- Configurable faults/anomalies with labeled ground truth.
- Data stored in Delta Tables for scalable analytics.

4. Agent Interaction
- Agents interact dynamically with user-selected assays and telemetry.
- Multi-agent orchestration enables reasoning across monitoring, troubleshooting, and reporting.
- Can extend to reinforcement learning agents or scenario planners for advanced gameplay.

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

Components:

| Component / Layer                     | Implementation / Module                 | Purpose / Notes                                                                                                   | Status / Notes for MVP                                                   |
| ------------------------------------- | --------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| **Fed-Batch Game Engine**             | `notebooks/synthetic_bioreactor.ipynb`  | Simulates fed-batch runs with biomass, substrate, product, DO, pH, HCP, titer. Adds noise, drift, and anomalies.  | Core MVP module. Generates synthetic data for all downstream components. |
|                                       | `backend/game_engine.py`                | Wraps simulation functions into reusable API endpoints; allows multiple runs and parameter sweeps.                | Optional for early MVP; enables integration with frontend/agents.        |
| **Delta Table / Data Lake**           | `notebooks/data_pipeline.ipynb`         | Saves telemetry and assay data in Delta tables on Databricks; supports batch & multi-run storage.                 | Essential for scalable analytics and agent queries.                      |
|                                       | `backend/data_api.py`                   | API endpoints to retrieve run data for UI or agent processing.                                                    | Optional for initial notebook-only MVP.                                  |
| **Anomaly Detection Layer**           | `notebooks/anomaly_detection.ipynb`     | Implements baseline analytics (z-score, moving average) and ML models (PyTorch / Isolation Forest).               | Phase 2; used to provide ground truth for agent reasoning.               |
|                                       | `backend/anomaly_models.py`             | Modular anomaly detection functions callable by API or agents.                                                    | Makes models reusable by copilot or dashboard.                           |
| **Multi-Agent Copilot**               | `agents/monitoring_agent.py`            | Observes telemetry and flags anomalies in real time or batch.                                                     | MVP version can use simple rules; later add ML/LLM reasoning.            |
|                                       | `agents/troubleshooting_agent.py`       | Interprets anomalies, recommends corrective actions, and optionally queries knowledge layer.                      | Key demonstration of AI reasoning and agent orchestration.               |
|                                       | `agents/reporting_agent.py`             | Summarizes run results and generates human-readable reports; can optionally produce PDF/Markdown.                 | Enhances portfolio presentation.                                         |
|                                       | `agents/fault_generator_agent.py`       | Optional: generates subtle composite faults for more challenging simulation scenarios.                            | Adds depth for Phase 3; not required for MVP.                            |
| **Knowledge Retrieval Layer**         | `agents/knowledge_agent.py`             | Queries curated OA bioprocess literature or vector DB; summarizes findings for user/agent hints.                  | Enables “Learn More” functionality; can start with small curated corpus. |
|                                       | `notebooks/knowledge_index.ipynb`       | Prepares vector embeddings (FAISS/Milvus/Weaviate) from abstracts/papers.                                         | Essential for reproducible, offline agent knowledge access.              |
| **Frontend Dashboard**                | `frontend/`                             | React + TypeScript + D3 dashboard; displays telemetry, assays, anomalies, agent messages, and “Learn More” links. | MVP can start with static plots; full interactivity comes later.         |
|                                       | `frontend/components/AgentChat.tsx`     | Chat interface for interacting with agents; requests hints, explanations, and report summaries.                   | Central for gamified learning experience.                                |
|                                       | `frontend/components/TelemetryPlot.tsx` | Dynamic plotting of time-series signals (X, S, P, DO, pH, HCP, titer).                                            | Core MVP visualization.                                                  |
| **Simulation Control / Gamification** | `backend/game_controller.py`            | Handles user-defined assay panels, applies faults, triggers agent hints, and manages simulation steps.            | Enables interactive gameplay mechanics.                                  |
| **Portfolio / Demo Runs**             | `notebooks/demo_runs.ipynb`             | Pre-generated runs demonstrating normal behavior, single faults, and composite faults for portfolio showcase.     | Useful for README, demo videos, or interactive showcase.                 |

