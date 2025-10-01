"""
run_simulation_workflow.py

Complete workflow integrating BioPilot simulation with REAL-TIME agentic control,
anomaly detection, and data persistence.
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
from agent_copilot import ( ExplainerAgent, LlamaRootCauseAgent)
from data_lake import BioreactorDataLake

# ---------------------------
# Workflow
# ---------------------------
class BioPilotWorkflow:
    def __init__(self, spark, config_dict: Dict, enable_agent: bool = True,
                 enable_anomaly_detection: bool = True, 
                 enable_agent_execution: bool = True):

        self.spark = spark
        self.config = config_dict
        self.run_id = str(uuid.uuid4())[:8]
        self.simulation = BioreactorSimulation(config_dict)

        self.data_lake = BioreactorDataLake()
        self.data_lake.create_schema_and_tables(spark)

        self.enable_anomaly_detection = enable_anomaly_detection
        print('anomaly detection enabled:', self.enable_anomaly_detection)
        if enable_anomaly_detection:
            anomaly_config = create_default_bioreactor_config()
            self.anomaly_detector = AnomalyDetectionEngine(anomaly_config)
            print('anomaly detection config:', anomaly_config)
        else:
            self.anomaly_detector = None

        self.enable_agent = enable_agent
        self.enable_agent_execution = enable_agent_execution
        if enable_agent:
            print('agentic analysis enabled.')
            self.agent = LlamaRootCauseAgent()
            self.explainer = ExplainerAgent()
        else:
            self.agent = None
            self.explainer = None

        self.all_anomaly_scores = []
        self.agent_budget = {'assays': 6}
        self.last_agent_check = 0.0
        self.agent_check_interval = 5.0  # Check every 5 hours

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
        print(f"Agent Execution: {'ENABLED' if self.enable_agent_execution else 'DISABLED'}")
        print(f"{'='*60}\n")

        dt = self.config['SIMULATION_PARAMS']['dt']
        duration = self.config['SIMULATION_PARAMS']['total_time']
        n_steps = int(duration / dt)
        
        true_history, observed_history = self.simulation.run(base_feed_rate)
        df_true = pd.DataFrame(true_history)
        df_obs = pd.DataFrame(observed_history)

        for idx, row in df_obs.iterrows():
            time = row['time']
            telemetry = {
                'X': row['X'],
                'S_glc': row['S_glc'],
                'P': row['P'],
                'DO': row['DO'],
                'pH': row['pH'],
                'time': time}
            
        print('TELEMETRY:', telemetry)

        # Anomaly detection pass
        if self.enable_anomaly_detection:
            print("\nRunning anomaly detection...")
            anomaly_results = self.anomaly_detector.detect_step(telemetry, time)
            print('ANOMALY RESULTS:', anomaly_results)
            self.all_anomaly_scores.extend(anomaly_results)
            
            anomaly_summary = self.anomaly_detector.get_anomaly_summary(
                self.all_anomaly_scores)
            print(f"Anomalies detected: {anomaly_summary['anomalies_detected']} / {anomaly_summary['total_checks']}")
            print(f"Anomaly rate: {anomaly_summary['anomaly_rate']:.2%}")
        
        print(f"\nSimulation complete!")
        print('type:', type(self.enable_anomaly_detection))
        if self.enable_anomaly_detection:
            print('getting anomaly summary...')
            anomaly_summary = self.anomaly_detector.get_anomaly_summary(self.all_anomaly_scores)
            print(f"Anomalies detected: {anomaly_summary['anomalies_detected']}")

        # explain agent troubleshooting details
        agent_explain = self.explainer.explain(true_history, 
                                                self.all_anomaly_scores)

        # save to data lake
        if save_to_lake:
            self.data_lake.save_telemetry(self.spark, self.run_id, true_history, is_observed=False)
            self.data_lake.save_telemetry(self.spark, self.run_id, observed_history, is_observed=True)
            if self.enable_anomaly_detection:
                self.data_lake.save_anomaly_scores(self.spark, self.run_id, self.all_anomaly_scores)
            final_titer_raw = df_true['P'].iloc[-1]
            final_titer = 0.0 if pd.isna(final_titer_raw) else float(final_titer_raw)
            self.data_lake.save_run_metadata(
                self.spark,
                run_id=self.run_id,
                config=self.config,
                scenario="standard",
                final_titer=final_titer,
                num_anomalies=len([a for a in self.all_anomaly_scores if a.is_anomaly]),
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
            'final_titer': df_true['P'].iloc[-1],
            'final_biomass': df_true['X'].iloc[-1],
            'num_anomalies': len([a for a in self.all_anomaly_scores if a.is_anomaly])
        }

        print(f"\nRun Complete! Final Titer: {summary['final_titer']:.2f}")
        return summary

# ---------------------------
# Visualization
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
                matching_obs = df_obs[df_obs['time'] == a.time]
                if not matching_obs.empty:
                    ax.scatter(a.time, matching_obs[sig].values[0], color='red', marker='x', s=100)

        ax.set_title(title)
        ax.set_xlabel('Time (h)')
        ax.legend()

    # Summary panel
    axes[-1].axis('off')
    summary_text = (
        f"Run: {results['run_id']}\n"
        f"Final Titer: {results['final_titer']:.2f} mg/mL\n"
        f"Anomalies: {results['num_anomalies']}\n")
    axes[-1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace', va='center')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()