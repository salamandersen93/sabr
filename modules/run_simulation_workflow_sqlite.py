import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
# run_simulation_workflow_sqlite.py
import numpy as np
import pandas as pd
from datetime import datetime
import uuid
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

from modules.config import (SIMULATION_PARAMS, INITIAL_STATE, KINETIC_PARAMS, 
                   REACTOR_PARAMS, SENSOR_PARAMS, FAULT_TEMPLATES)
from modules.models import BioreactorSimulation
from modules.anomaly_detection import (AnomalyDetectionEngine, create_default_bioreactor_config)
from modules.agent_copilot import (ExplainerAgent)
from modules.data_lake_sqlite import BioreactorDataLakeSQLite

class SABRWorkflow:
    def __init__(self, config_dict: Dict, enable_agent: bool = True,
                 enable_anomaly_detection: bool = True, 
                 enable_agent_execution: bool = True):

        self.config = config_dict
        self.run_id = str(uuid.uuid4())[:8]
        self.simulation = BioreactorSimulation(config_dict)

        # Use SQLite-based data lake
        self.data_lake = BioreactorDataLakeSQLite(db_path="sabr_db.sqlite")

        self.enable_anomaly_detection = enable_anomaly_detection
        if enable_anomaly_detection:
            anomaly_config = self._create_tuned_anomaly_config()
            self.anomaly_detector = AnomalyDetectionEngine(anomaly_config)
        else:
            self.anomaly_detector = None

        self.enable_agent = enable_agent
        self.enable_agent_execution = enable_agent_execution
        self.explainer = ExplainerAgent() if enable_agent else None

        self.all_anomaly_scores = []
        self.last_agent_check = 0.0
        self.agent_check_interval = 5.0

    def _create_tuned_anomaly_config(self) -> Dict:
        return {
            'X': {'moving_window': {'window_size': 30, 'threshold_sigma': 3.5},
                  'rate_of_change': {'max_rate': 0.15}},
            'S_glc': {'moving_window': {'window_size': 25, 'threshold_sigma': 3.0},
                      'rate_of_change': {'max_rate': 0.5}},
            'DO': {'moving_window': {'window_size': 20, 'threshold_sigma': 2.5},
                   'rate_of_change': {'max_rate': 20.0}},
            'pH': {'moving_window': {'window_size': 25, 'threshold_sigma': 2.0},
                   'rate_of_change': {'max_rate': 0.15}},
            'P': {'moving_window': {'window_size': 40, 'threshold_sigma': 3.5},
                  'rate_of_change': {'max_rate': 0.08}},
            'correlations': {
                ('X', 'DO'): {'expected': -0.6, 'tolerance': 0.45},
                ('X', 'S_glc'): {'expected': -0.5, 'tolerance': 0.45},
                ('X', 'P'): {'expected': 0.8, 'tolerance': 0.3}
            }
        }

    def inject_scenario_faults(self, scenario: str = "overfeed"):
        if scenario in self.simulation.fault_manager.fault_templates:
            start_time = self.simulation.fault_manager.fault_templates[scenario].get('start_h', 20.0)
            self.simulation.fault_manager.activate_fault(scenario, start_time)
            fault_info = self.simulation.fault_manager.fault_templates[scenario]
            self.data_lake.save_fault_log(
                run_id=self.run_id,
                fault_id=f"{scenario}_{start_time}",
                fault_type=scenario,
                start_time=start_time,
                duration=fault_info.get('duration_h', 2.0),
                parameters=fault_info)

    def run_with_monitoring(self, base_feed_rate: float = 0.1, 
                           save_to_lake: bool = True,
                           verbose: bool = True) -> Dict:
        start_timestamp = datetime.now()
        dt = self.config['SIMULATION_PARAMS']['dt']
        duration = self.config['SIMULATION_PARAMS']['total_time']
        
        true_history, observed_history = self.simulation.run(base_feed_rate)
        df_true = pd.DataFrame(true_history)
        df_obs = pd.DataFrame(observed_history)

        anomaly_counts_by_step = []

        for idx, row in df_obs.iterrows():
            time = row['time']
            telemetry = {sig: row[sig] for sig in ['X', 'S_glc', 'P', 'DO', 'pH']}

            if self.enable_anomaly_detection:
                anomaly_results = self.anomaly_detector.detect_step(telemetry, time)
                self.all_anomaly_scores.extend(anomaly_results)
                step_anomalies = sum(1 for a in anomaly_results if a.is_anomaly)
                anomaly_counts_by_step.append(step_anomalies)
                if verbose and step_anomalies > 0:
                    print(f"t={time:.1f}h: {step_anomalies} anomalies detected")
                    for a in anomaly_results:
                        if a.is_anomaly:
                            print(f"  {a.signal}: {a.method} score={a.score:.2f}")

        agent_explain = None
        if self.enable_agent:
            agent_explain = self.explainer.explain(true_history, self.all_anomaly_scores)

        if save_to_lake:
            self.data_lake.save_telemetry(self.run_id, true_history, is_observed=False)
            self.data_lake.save_telemetry(self.run_id, observed_history, is_observed=True)
            if self.enable_anomaly_detection:
                self.data_lake.save_anomaly_scores(self.run_id, self.all_anomaly_scores)
            
            final_titer = float(df_true['P'].iloc[-1])
            self.data_lake.save_run_metadata(
                run_id=self.run_id,
                config=self.config,
                scenario="standard",
                final_titer=final_titer,
                num_anomalies=len([a for a in self.all_anomaly_scores if a.is_anomaly]),
                success=final_titer > 5.0,
                score=final_titer * 10.0,
                start_time=start_timestamp,
                end_time=datetime.now())

        summary = {
            'run_id': self.run_id,
            'true_history': df_true,
            'observed_history': df_obs,
            'anomaly_scores': self.all_anomaly_scores,
            'anomaly_counts_by_step': anomaly_counts_by_step,
            'final_titer': df_true['P'].iloc[-1],
            'final_biomass': df_true['X'].iloc[-1],
            'num_anomalies': len([a for a in self.all_anomaly_scores if a.is_anomaly]),
            'agent_explain': agent_explain
        }

        return summary