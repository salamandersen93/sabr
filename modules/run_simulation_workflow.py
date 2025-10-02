"""
Fixed run_simulation_workflow.py - Corrected anomaly detection integration
"""

import numpy as np
import pandas as pd
from datetime import datetime
import uuid
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

from config import (SIMULATION_PARAMS, INITIAL_STATE, KINETIC_PARAMS, 
                   REACTOR_PARAMS, SENSOR_PARAMS, FAULT_TEMPLATES)
from models import BioreactorSimulation
from anomaly_detection import (AnomalyDetectionEngine, create_default_bioreactor_config)
from agent_copilot import (ExplainerAgent)
from data_lake import BioreactorDataLake


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
            anomaly_config = self._create_tuned_anomaly_config()  # FIXED: Use tuned config
            self.anomaly_detector = AnomalyDetectionEngine(anomaly_config)
            print('anomaly detection config:', anomaly_config)
        else:
            self.anomaly_detector = None

        self.enable_agent = enable_agent
        self.enable_agent_execution = enable_agent_execution
        if enable_agent:
            print('agentic analysis enabled.')
            self.explainer = ExplainerAgent()
        else:
            self.agent = None
            self.explainer = None

        self.all_anomaly_scores = []
        self.last_agent_check = 0.0
        self.agent_check_interval = 5.0

    def _create_tuned_anomaly_config(self) -> Dict:
        """
        Create anomaly detection config tuned to actual bioreactor dynamics.
        
        CRITICAL: Product (P) is in g/L, not mg/mL!
        """
        return {
            'X': {
                'moving_window': {'window_size': 30, 'threshold_sigma': 3.5},
                'rate_of_change': {'max_rate': 0.15}  # g/L per hour (realistic for exponential growth)
            },
            'S_glc': {
                'moving_window': {'window_size': 25, 'threshold_sigma': 3.0},
                'rate_of_change': {'max_rate': 0.5}  # g/L per hour (consumption + feed)
            },
            'DO': {
                'moving_window': {'window_size': 20, 'threshold_sigma': 2.5},
                'rate_of_change': {'max_rate': 20.0}  # %DO per hour (can drop fast)
            },
            'pH': {
                'moving_window': {'window_size': 25, 'threshold_sigma': 2.0},
                'rate_of_change': {'max_rate': 0.15}  # pH units per hour
            },
            'P': {
                'moving_window': {'window_size': 40, 'threshold_sigma': 3.5},
                'rate_of_change': {'max_rate': 0.08}  # g/L per hour (FIXED: was mg/mL)
            },
            'correlations': {
                ('X', 'DO'): {'expected': -0.6, 'tolerance': 0.45},
                ('X', 'S_glc'): {'expected': -0.5, 'tolerance': 0.45},
                ('X', 'P'): {'expected': 0.8, 'tolerance': 0.3}  # Product should correlate with biomass
            }
        }

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
                parameters=fault_info)

    def run_with_monitoring(self, base_feed_rate: float = 0.1, 
                           save_to_lake: bool = True,
                           verbose: bool = True) -> Dict:
        start_timestamp = datetime.now()
        print(f"\n{'='*60}")
        print(f"BioPilot Simulation Run: {self.run_id}")
        print(f"Agent Execution: {'ENABLED' if self.enable_agent_execution else 'DISABLED'}")
        print(f"Anomaly Detection: {'ENABLED' if self.enable_anomaly_detection else 'DISABLED'}")
        print(f"{'='*60}\n")

        dt = self.config['SIMULATION_PARAMS']['dt']
        duration = self.config['SIMULATION_PARAMS']['total_time']
        
        # Run simulation
        true_history, observed_history = self.simulation.run(base_feed_rate)
        df_true = pd.DataFrame(true_history)
        df_obs = pd.DataFrame(observed_history)

        # FIXED: Anomaly detection INSIDE the loop
        print("Processing telemetry and detecting anomalies...")
        anomaly_counts_by_step = []
        
        for idx, row in df_obs.iterrows():
            time = row['time']
            telemetry = {
                'X': row['X'],
                'S_glc': row['S_glc'],
                'P': row['P'],
                'DO': row['DO'],
                'pH': row['pH']
            }
            
            # Run anomaly detection for THIS timestep
            if self.enable_anomaly_detection:
                anomaly_results = self.anomaly_detector.detect_step(telemetry, time)
                self.all_anomaly_scores.extend(anomaly_results)
                
                # Count anomalies at this step
                step_anomalies = sum(1 for a in anomaly_results if a.is_anomaly)
                anomaly_counts_by_step.append(step_anomalies)
                
                # Verbose output for debugging
                if verbose and step_anomalies > 0:
                    print(f"  t={time:.1f}h: {step_anomalies} anomalies detected")
                    for a in anomaly_results:
                        if a.is_anomaly:
                            print(f"    - {a.signal}: {a.method} score={a.score:.2f}")
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"Simulation complete!")
        
        if self.enable_anomaly_detection:
            anomaly_summary = self.anomaly_detector.get_anomaly_summary(
                self.all_anomaly_scores)
            print(f"\nAnomaly Detection Summary:")
            print(f"  Total checks: {anomaly_summary['total_checks']}")
            print(f"  Anomalies detected: {anomaly_summary['anomalies_detected']}")
            print(f"  Anomaly rate: {anomaly_summary['anomaly_rate']:.2%}")
            
            if anomaly_summary['by_signal']:
                print(f"\n  By Signal:")
                for signal, count in anomaly_summary['by_signal'].items():
                    print(f"    {signal}: {count}")
            
            if anomaly_summary['by_method']:
                print(f"\n  By Method:")
                for method, count in anomaly_summary['by_method'].items():
                    print(f"    {method}: {count}")

        # Agent explanation
        if self.enable_agent:
            print("\nGenerating agent explanation...")
            agent_explain = self.explainer.explain(true_history, self.all_anomaly_scores)

        # Save to data lake
        if save_to_lake:
            print("\nSaving to data lake...")
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

        print(f"\n{'='*60}")
        print(f"Run Complete!")
        print(f"  Final Titer: {summary['final_titer']:.2f} g/L")
        print(f"  Final Biomass: {summary['final_biomass']:.2f} g/L")
        print(f"{'='*60}\n")
        
        return summary


def visualize_run(results: Dict, save_path: Optional[str] = None):
    """Enhanced visualization with anomaly highlighting."""
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
                # Get observed values at anomaly times
                anomaly_mask = df_obs['time'].isin(anomaly_times)
                ax.scatter(df_obs.loc[anomaly_mask, 'time'], 
                          df_obs.loc[anomaly_mask, sig],
                          color='red', marker='x', s=100, 
                          label=f'Anomalies ({len(anomaly_times)})', 
                          zorder=5)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (h)')
        ax.legend()
        ax.grid(alpha=0.3)

    # Enhanced summary panel
    axes[-1].axis('off')
    total_anomalies = results['num_anomalies']
    anomaly_by_signal = {}
    for a in results['anomaly_scores']:
        if a.is_anomaly:
            anomaly_by_signal[a.signal] = anomaly_by_signal.get(a.signal, 0) + 1
    
    summary_text = (
        f"Run ID: {results['run_id']}\n\n"
        f"Final Metrics:\n"
        f"  Titer: {results['final_titer']:.2f} g/L\n"
        f"  Biomass: {results['final_biomass']:.2f} g/L\n\n"
        f"Anomalies: {total_anomalies}\n"
    )
    
    if anomaly_by_signal:
        summary_text += "By Signal:\n"
        for sig, count in sorted(anomaly_by_signal.items()):
            summary_text += f"  {sig}: {count}\n"
    
    axes[-1].text(0.1, 0.5, summary_text, fontsize=10, family='monospace', 
                  va='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


# DIAGNOSTIC HELPER
def diagnose_anomaly_detection(results: Dict):
    """
    Print detailed diagnostics about why anomalies were/weren't detected.
    """
    print("\n" + "="*60)
    print("ANOMALY DETECTION DIAGNOSTICS")
    print("="*60)
    
    df_obs = results['observed_history']
    
    # Check data ranges
    print("\nData Ranges (Observed):")
    for sig in ['X', 'S_glc', 'P', 'DO', 'pH']:
        print(f"  {sig}: [{df_obs[sig].min():.3f}, {df_obs[sig].max():.3f}]")
    
    # Check rates of change
    print("\nMax Rates of Change:")
    for sig in ['X', 'S_glc', 'P', 'DO', 'pH']:
        diff = df_obs[sig].diff().abs()
        dt = df_obs['time'].diff()
        rates = (diff / dt).dropna()
        if len(rates) > 0:
            print(f"  {sig}: {rates.max():.3f} units/h")
    
    # Anomaly breakdown
    if results['anomaly_scores']:
        print(f"\nTotal Anomaly Checks: {len(results['anomaly_scores'])}")
        print(f"Anomalies Detected: {results['num_anomalies']}")
        
        methods = {}
        signals = {}
        for a in results['anomaly_scores']:
            methods[a.method] = methods.get(a.method, 0) + (1 if a.is_anomaly else 0)
            signals[a.signal] = signals.get(a.signal, 0) + (1 if a.is_anomaly else 0)
        
        print("\nDetections by Method:")
        for method, count in methods.items():
            print(f"  {method}: {count}")
        
        print("\nDetections by Signal:")
        for signal, count in sorted(signals.items()):
            if count > 0:
                print(f"  {signal}: {count}")
    
    print("="*60 + "\n")


# UNIT CONSISTENCY CHECK
def verify_units_consistency():
    """
    Helper to verify units are consistent across the codebase.
    Call this during initialization to catch unit mismatches.
    """
    print("\nUnit Consistency Check:")
    print("  X (Biomass): g/L")
    print("  S_glc (Glucose): g/L")
    print("  P (Product): g/L  ⚠️  NOTE: NOT mg/mL!")
    print("  DO (Dissolved O2): %")
    print("  pH: pH units")
    print("\nTypical ranges:")
    print("  X: 0-50 g/L (CHO culture)")
    print("  S_glc: 0-20 g/L")
    print("  P: 0-10 g/L (1-10 g/L typical for mAb)")
    print("  DO: 0-100%")
    print("  pH: 6.5-7.6")
    print()