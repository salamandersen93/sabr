"""
run_simulation_workflow.py

Complete workflow integrating BioPilot simulation, anomaly detection,
true agentic analysis (via Databricks Llama endpoint), 
and data persistence.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import uuid
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import json
import mlflow.deployments

# import components
from config import (SIMULATION_PARAMS, INITIAL_STATE, KINETIC_PARAMS, REACTOR_PARAMS, SENSOR_PARAMS, FAULT_TEMPLATES)
from models import BioreactorSimulation
from anomaly_detection import (AnomalyDetectionEngine, create_default_bioreactor_config)
from agent_copilot import AgentObservation  # still using the observation structure
from data_lake import BioreactorDataLake


# ---------------------------
# True Agent wrapper (Databricks Llama)
# ---------------------------
class LlamaRootCauseAgent:
    def __init__(self, endpoint="databricks-meta-llama-3-3-70b-instruct"):
        self.client = mlflow.deployments.get_deploy_client("databricks")
        self.endpoint = endpoint

    def analyze(self, observation) -> dict:
        """Ask the LLM to analyze telemetry and anomalies."""
        messages = [
            {"role": "system", "content": "You are a bioprocess expert. Explain anomalies in bioreactor telemetry and recommend operator actions."},
            {"role": "user", "content": f"""
            Telemetry: {observation.telemetry}
            Anomalies: {[str(a.__dict__) for a in observation.recent_anomalies]}
            Time: {observation.time} h
            Assay budget remaining: {observation.available_budget['assays']}

            Respond ONLY in JSON with fields:
            root_cause: str
            recommended_action: str
            priority: int (1=low, 5=critical)
            rationale: str
            """}
        ]

        response = self.client.predict(
            endpoint=self.endpoint,
            inputs={"messages": messages, "temperature": 0.2, "max_tokens": 300}
        )

        try:
            return json.loads(response["choices"][0]["message"]["content"])
        except Exception:
            return {
                "root_cause": "Unknown",
                "recommended_action": "No action",
                "priority": 1,
                "rationale": "Model output not parsed as JSON."
            }


# ---------------------------
# Workflow
# ---------------------------
class BioPilotWorkflow:
    def __init__(self, spark, config_dict: Dict, enable_agent: bool = True,
                 enable_anomaly_detection: bool = True):

        self.spark = spark
        self.config = config_dict
        self.run_id = str(uuid.uuid4())[:8]
        self.simulation = BioreactorSimulation(config_dict)

        self.data_lake = BioreactorDataLake()
        self.data_lake.create_schema_and_tables(spark)

        self.enable_anomaly_detection = enable_anomaly_detection
        if enable_anomaly_detection:
            anomaly_config = create_default_bioreactor_config()
            self.anomaly_detector = AnomalyDetectionEngine(anomaly_config)
        else:
            self.anomaly_detector = None

        self.enable_agent = enable_agent
        if enable_agent:
            self.agent = LlamaRootCauseAgent()
        else:
            self.agent = None

        self.all_anomaly_scores = []
        self.agent_budget = {'assays': 6}

    def inject_scenario_faults(self, scenario: str = "overfeed"):
        if scenario in self.simulation.fault_manager.fault_templates:
            start_time = self.simulation.fault_manager.fault_templates[scenario].get('start_h', 20.0)
            self.simulation.fault_manager.activate_fault(scenario, start_time)
            fault_info = self.simulation.fault_manager.fault_templates[scenario]
            self.data_lake.save_fault_log(
                self.spark,
                run_id=self.run_id,
                fault_id=f"{scenario}_{start_time}",
                fault_type=scenario,
                start_time=start_time,
                duration=fault_info.get('duration_h', 2.0),
                parameters=fault_info
            )

    def run_with_monitoring(self, base_feed_rate: float = 0.1, save_to_lake: bool = True) -> Dict:
        start_timestamp = datetime.now()
        print(f"\n{'='*60}")
        print(f"BioPilot Simulation Run: {self.run_id}")
        print(f"{'='*60}\n")

        # run simulation
        true_history, observed_history = self.simulation.run(base_feed_rate)
        df_true = pd.DataFrame(true_history)
        df_obs = pd.DataFrame(observed_history)

        # anomaly detection
        if self.enable_anomaly_detection:
            for _, row in df_obs.iterrows():
                signals = {sig: row[sig] for sig in ['X', 'S_glc', 'P', 'DO', 'pH']}
                anomaly_results = self.anomaly_detector.detect_step(signals, row['time'])
                self.all_anomaly_scores.extend(anomaly_results)
            anomaly_summary = self.anomaly_detector.get_anomaly_summary(self.all_anomaly_scores)
            print(f"Anomalies detected: {anomaly_summary['anomalies_detected']}")

        # agent loop
        agent_actions = []
        if self.enable_agent:
            print("\nAgent analyzing telemetry...")
            window_size = 5
            for _, row in df_obs.iterrows():
                time = row['time']
                telemetry = {sig: row[sig] for sig in ['X', 'S_glc', 'P', 'DO', 'pH']}
                telemetry['time'] = time
                recent_anomalies = [a for a in self.all_anomaly_scores if abs(a.time - time) < window_size * self.config['SIMULATION_PARAMS']['dt']]
                observation = AgentObservation(
                    time=time,
                    telemetry=telemetry,
                    recent_anomalies=recent_anomalies,
                    recent_actions=agent_actions[-5:] if agent_actions else [],
                    available_budget=self.agent_budget.copy()
                )
                agent_output = self.agent.analyze(observation)

                action = {
                    "time": time,
                    "root_cause": agent_output["root_cause"],
                    "recommended_action": agent_output["recommended_action"],
                    "priority": agent_output["priority"],
                    "rationale": agent_output["rationale"]
                }

                if action["priority"] >= 4:
                    agent_actions.append(action)
                    if "assay" in action["recommended_action"].lower() and self.agent_budget['assays'] > 0:
                        self.agent_budget['assays'] -= 1

            print(f"Agent actions taken: {len(agent_actions)}")

        # save to data lake
        if save_to_lake:
            self.data_lake.save_telemetry(self.spark, self.run_id, true_history, is_observed=False)
            self.data_lake.save_telemetry(self.spark, self.run_id, observed_history, is_observed=True)
            if self.enable_anomaly_detection:
                self.data_lake.save_anomaly_scores(self.spark, self.run_id, self.all_anomaly_scores)
            if self.enable_agent and agent_actions:
                for idx, action in enumerate(agent_actions):
                    self.data_lake.save_agent_action(
                        self.spark,
                        run_id=self.run_id,
                        action_id=f"agent_{idx}_{action['time']}",
                        time=action['time'],
                        action_type=action['recommended_action'],
                        parameters={"root_cause": action['root_cause']},
                        rationale=action['rationale']
                    )
            final_titer_raw = df_true['P'].iloc[-1]
            final_titer = 0.0 if pd.isna(final_titer_raw) else float(final_titer_raw)
            self.data_lake.save_run_metadata(
                self.spark,
                run_id=self.run_id,
                config=self.config,
                scenario="standard",
                final_titer=final_titer,
                num_anomalies=len([a for a in self.all_anomaly_scores if a.is_anomaly]),
                num_actions=len(agent_actions),
                success=final_titer > 5.0,
                score=final_titer * 10.0,
                start_time=start_timestamp,
                end_time=datetime.now()
            )

        summary = {
            'run_id': self.run_id,
            'true_history': df_true,
            'observed_history': df_obs,
            'anomaly_scores': self.all_anomaly_scores,
            'agent_actions': agent_actions,
            'final_titer': df_true['P'].iloc[-1],
            'final_biomass': df_true['X'].iloc[-1],
            'num_anomalies': len([a for a in self.all_anomaly_scores if a.is_anomaly]),
            'num_actions': len(agent_actions)
        }

        print(f"Run Complete! Final Titer: {summary['final_titer']:.2f}")
        return summary


# ---------------------------
# Visualization (with agent explanations)
# ---------------------------
def visualize_run(results: Dict, save_path: Optional[str] = None):
    df_true = results['true_history']
    df_obs = results['observed_history']

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    signals = ['X', 'S_glc', 'P', 'DO', 'pH']
    titles = ['Biomass [g/L]', 'Glucose [g/L]', 'Product Titer [mg/mL]', 'Dissolved Oxygen [%]', 'pH']

    for i, (sig, title) in enumerate(zip(signals, titles)):
        ax = axes[i]
        ax.plot(df_true['time'], df_true[sig], label='True', linewidth=2)
        ax.plot(df_obs['time'], df_obs[sig], label='Observed', linestyle='--')

        # anomalies
        if results['anomaly_scores']:
            sig_anomalies = [a for a in results['anomaly_scores'] if a.signal == sig and a.is_anomaly]
            for a in sig_anomalies:
                ax.scatter(a.time, df_obs.loc[df_obs['time'] == a.time, sig], color='red', marker='x')
                ax.annotate("Anomaly", (a.time, df_obs.loc[df_obs['time'] == a.time, sig].values[0]),
                            xytext=(5, 5), textcoords="offset points", fontsize=8, color='red')

        # agent actions
        if results['agent_actions']:
            for action in results['agent_actions']:
                ax.axvline(action['time'], color='green', linestyle=':')
                ax.annotate(action['root_cause'], xy=(action['time'], df_obs[sig].mean()),
                            xytext=(5, 10), textcoords="offset points", fontsize=8, color='green')

        ax.set_title(title)
        ax.legend()

    axes[-1].axis('off')
    axes[-1].text(0.1, 0.5, f"Run {results['run_id']}\nFinal Titer {results['final_titer']:.2f}",
                  fontsize=11, family='monospace')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
