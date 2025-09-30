"""
run_simulation_workflow.py

Complete workflow integrating all BioPilot components.
Demonstrates end-to-end execution with fault injection, anomaly detection, 
agent actions, and data persistence.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import uuid
from typing import Dict, List, Optional
import math

# Import all components
from config import (
    SIMULATION_PARAMS, INITIAL_STATE, KINETIC_PARAMS, 
    REACTOR_PARAMS, SENSOR_PARAMS, FAULT_TEMPLATES
)
from models import BioreactorSimulation, FaultManager
from anomaly_detection import (
    AnomalyDetectionEngine, 
    create_default_bioreactor_config
)
from agent_copilot import (
    MultiAgentCopilot, 
    create_default_copilot_config,
    AgentObservation
)
from data_lake import BioreactorDataLake


class BioPilotWorkflow:
    """Orchestrates complete simulation workflow with all components."""
    
    def __init__(self, spark, config_dict: Dict, 
                 enable_agent: bool = True,
                 enable_anomaly_detection: bool = True):
        """
        Initialize workflow with configuration.
        
        Args:
            spark: Spark session for Databricks
            config_dict: Complete configuration dictionary
            enable_agent: Whether to use agent copilot
            enable_anomaly_detection: Whether to run anomaly detection
        """
        self.spark = spark
        self.config = config_dict
        self.run_id = str(uuid.uuid4())[:8]
        
        # Initialize components
        self.simulation = BioreactorSimulation(config_dict)
        
        # Data lake (filesystem-backed Delta tables)
        self.data_lake = BioreactorDataLake(base_namespace="biopilot")
        self.data_lake.initialize_schema(spark)
        
        # Anomaly detection
        self.enable_anomaly_detection = enable_anomaly_detection
        if enable_anomaly_detection:
            anomaly_config = create_default_bioreactor_config()
            self.anomaly_detector = AnomalyDetectionEngine(anomaly_config)
        else:
            self.anomaly_detector = None
        
        # Agent copilot
        self.enable_agent = enable_agent
        if enable_agent:
            copilot_config = create_default_copilot_config()
            self.copilot = MultiAgentCopilot(copilot_config, use_predictive=False)
        else:
            self.copilot = None
        
        # Runtime state
        self.all_anomaly_scores = []
        self.agent_budget = {'assays': 6}  # From GAME_PARAMS
    
    def inject_scenario_faults(self, scenario: str = "overfeed"):
        """
        Configure faults for specific scenario.
        
        Args:
            scenario: Scenario name (overfeed, underfeed, DO_drop, etc.)
        """
        if scenario in self.simulation.fault_manager.fault_templates:
            start_time = self.simulation.fault_manager.fault_templates[scenario].get('start_h', 20.0)
            self.simulation.fault_manager.activate_fault(scenario, start_time)
            
            # Log fault injection
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
    
    def run_with_monitoring(self, base_feed_rate: float = 0.1,
                           save_to_lake: bool = True) -> Dict:
        """
        Execute simulation with real-time monitoring and agent intervention.
        
        Returns:
            Summary dictionary with results
        """
        start_timestamp = datetime.now()
        
        print(f"\n{'='*60}")
        print(f"BioPilot Simulation Run: {self.run_id}")
        print(f"{'='*60}\n")
        
        # Run base simulation
        print("Running bioreactor simulation...")
        true_history, observed_history = self.simulation.run(base_feed_rate)
        
        print(f"Simulation complete: {len(true_history)} timesteps")
        
        # Convert to DataFrames
        df_true = pd.DataFrame(true_history)
        df_obs = pd.DataFrame(observed_history)
        
        # Anomaly detection pass
        if self.enable_anomaly_detection:
            print("\nRunning anomaly detection...")
            for idx, row in df_obs.iterrows():
                signals = {
                    'X': row['X'],
                    'S_glc': row['S_glc'],
                    'P': row['P'],
                    'DO': row['DO'],
                    'pH': row['pH']
                }
                time = row['time']
                
                anomaly_results = self.anomaly_detector.detect_step(signals, time)
                self.all_anomaly_scores.extend(anomaly_results)
            
            anomaly_summary = self.anomaly_detector.get_anomaly_summary(
                self.all_anomaly_scores
            )
            print(f"Anomalies detected: {anomaly_summary['anomalies_detected']} / {anomaly_summary['total_checks']}")
            print(f"Anomaly rate: {anomaly_summary['anomaly_rate']:.2%}")
        
        # Agent copilot pass
        agent_actions = []
        if self.enable_agent:
            print("\nAgent copilot analyzing telemetry...")
            
            # Window for recent anomalies
            window_size = 5
            
            for idx, row in df_obs.iterrows():
                time = row['time']
                telemetry = {
                    'X': row['X'],
                    'S_glc': row['S_glc'],
                    'P': row['P'],
                    'DO': row['DO'],
                    'pH': row['pH'],
                    'time': time
                }
                
                # Get recent anomalies for this timestep
                recent_anomalies = [
                    a for a in self.all_anomaly_scores
                    if abs(a.time - time) < window_size * self.config['SIMULATION_PARAMS']['dt']
                ]
                
                # Create observation
                observation = AgentObservation(
                    time=time,
                    telemetry=telemetry,
                    recent_anomalies=recent_anomalies,
                    recent_actions=agent_actions[-5:] if agent_actions else [],
                    available_budget=self.agent_budget.copy()
                )
                
                # Get agent recommendations
                actions = self.copilot.process_observation(observation)
                
                # Execute high-priority actions
                for action in actions:
                    if action.priority >= 4:  # Execute critical actions
                        agent_actions.append(action)
                        
                        # Update budget if assay requested
                        if action.action_type.value == 'request_assay':
                            if self.agent_budget['assays'] > 0:
                                self.agent_budget['assays'] -= 1
            
            print(f"Agent actions taken: {len(agent_actions)}")
            
            # Generate agent report
            agent_report = self.copilot.generate_report()
            print("\n" + agent_report)
        else:
            agent_report = None
        
        # Save to data lake
        if save_to_lake:
            print("\nSaving results to data lake...")
            
            # Save telemetry (both true and observed)
            self.data_lake.save_telemetry(
                self.spark, self.run_id, true_history, 
                is_observed=False, batch_id=0
            )
            self.data_lake.save_telemetry(
                self.spark, self.run_id, observed_history,
                is_observed=True, batch_id=0
            )
            
            # Save anomaly scores
            if self.enable_anomaly_detection:
                self.data_lake.save_anomaly_scores(
                    self.spark, self.run_id, self.all_anomaly_scores
                )
            
            # Save agent actions
            if self.enable_agent and agent_actions:
                for action in agent_actions:
                    action_id = f"{action.action_type.value}_{action.time}"
                    self.data_lake.save_agent_action(
                        self.spark,
                        run_id=self.run_id,
                        action_id=action_id,
                        time=action.time,
                        action_type=action.action_type.value,
                        parameters=action.parameters,
                        rationale=action.rationale
                    )
            
            # Save run metadata
            end_timestamp = datetime.now()

            final_titer_raw = df_true['P'].iloc[-1]
            final_titer = 0.0 if (final_titer_raw is None or pd.isna(final_titer_raw)) else float(final_titer_raw)

            
            self.data_lake.save_run_metadata(
                self.spark,
                run_id=self.run_id,
                config=self.config,
                scenario="standard",
                final_titer=final_titer,
                num_anomalies=len([a for a in self.all_anomaly_scores if a.is_anomaly]),
                num_actions=len(agent_actions),
                success=final_titer > 5.0,  # Arbitrary success threshold
                score=final_titer * 10.0,  # Simple scoring
                start_time=start_timestamp,
                end_time=end_timestamp
            )
            
            print(f"Data saved to data lake with run_id: {self.run_id}")
        
        # Compile summary
        summary = {
            'run_id': self.run_id,
            'true_history': df_true,
            'observed_history': df_obs,
            'anomaly_scores': self.all_anomaly_scores,
            'agent_actions': agent_actions,
            'final_titer': df_true['P'].iloc[-1],
            'final_biomass': df_true['X'].iloc[-1],
            'num_anomalies': len([a for a in self.all_anomaly_scores if a.is_anomaly]),
            'num_actions': len(agent_actions),
            'agent_report': agent_report
        }
        
        print(f"\n{'='*60}")
        print(f"Run Complete!")
        print(f"Final Titer: {summary['final_titer']:.2f} mg/mL")
        print(f"Final Biomass: {summary['final_biomass']:.2f} g/L")
        print(f"{'='*60}\n")
        
        return summary

def run_example_workflow(spark):
    """
    Example usage of complete BioPilot workflow.
    """
    
    # Assemble configuration
    config = {
        'SIMULATION_PARAMS': SIMULATION_PARAMS,
        'INITIAL_STATE': INITIAL_STATE,
        'KINETIC_PARAMS': KINETIC_PARAMS,
        'REACTOR_PARAMS': REACTOR_PARAMS,
        'SENSOR_PARAMS': SENSOR_PARAMS,
        'FAULT_TEMPLATES': FAULT_TEMPLATES
    }
    
    # Initialize workflow
    workflow = BioPilotWorkflow(
        spark=spark,
        config_dict=config,
        enable_agent=True,
        enable_anomaly_detection=True
    )
    
    # Inject a fault scenario
    workflow.inject_scenario_faults(scenario="overfeed")
    
    # Run simulation with monitoring
    results = workflow.run_with_monitoring(
        base_feed_rate=0.1,
        save_to_lake=True
    )
    
    return results, workflow


def run_batch_scenarios(spark, scenarios: List[str], 
                       num_replicates: int = 3) -> pd.DataFrame:
    """
    Run multiple scenario replicates for analysis.
    
    Args:
        spark: Spark session
        scenarios: List of scenario names to test
        num_replicates: Number of replicates per scenario
    
    Returns:
        DataFrame with summary statistics for all runs
    """
    
    config = {
        'SIMULATION_PARAMS': SIMULATION_PARAMS,
        'INITIAL_STATE': INITIAL_STATE,
        'KINETIC_PARAMS': KINETIC_PARAMS,
        'REACTOR_PARAMS': REACTOR_PARAMS,
        'SENSOR_PARAMS': SENSOR_PARAMS,
        'FAULT_TEMPLATES': FAULT_TEMPLATES
    }
    
    all_results = []
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Running scenario: {scenario}")
        print(f"{'='*60}")
        
        for replicate in range(num_replicates):
            print(f"\nReplicate {replicate + 1}/{num_replicates}")
            
            # Vary random seed for each replicate
            config['SIMULATION_PARAMS']['random_seed'] = 42 + replicate
            
            workflow = BioPilotWorkflow(
                spark=spark,
                config_dict=config,
                enable_agent=True,
                enable_anomaly_detection=True
            )
            
            # Inject scenario fault
            if scenario != "baseline":
                workflow.inject_scenario_faults(scenario=scenario)
            
            # Run
            results = workflow.run_with_monitoring(
                base_feed_rate=0.1,
                save_to_lake=True
            )
            
            # Collect summary stats
            all_results.append({
                'run_id': results['run_id'],
                'scenario': scenario,
                'replicate': replicate,
                'final_titer': results['final_titer'],
                'final_biomass': results['final_biomass'],
                'num_anomalies': results['num_anomalies'],
                'num_actions': results['num_actions']
            })
    
    # Create summary DataFrame
    df_summary = pd.DataFrame(all_results)
    
    # Print aggregate statistics
    print("\n" + "="*60)
    print("BATCH RUN SUMMARY")
    print("="*60)
    print(df_summary.groupby('scenario').agg({
        'final_titer': ['mean', 'std'],
        'num_anomalies': 'mean',
        'num_actions': 'mean'
    }))
    
    return df_summary


def visualize_run(results: Dict, save_path: Optional[str] = None):
    """
    Create visualization dashboard for a single run.
    """
    import matplotlib.pyplot as plt
    
    df_true = results['true_history']
    df_obs = results['observed_history']
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    signals = ['X', 'S_glc', 'P', 'DO', 'pH']
    titles = [
        'Biomass [g/L]',
        'Glucose [g/L]',
        'Product Titer [mg/mL]',
        'Dissolved Oxygen [%]',
        'pH'
    ]
    
    for i, (sig, title) in enumerate(zip(signals, titles)):
        ax = axes[i]
        
        # Plot true and observed
        ax.plot(df_true['time'], df_true[sig], 
               label='True', linewidth=2, alpha=0.8)
        ax.plot(df_obs['time'], df_obs[sig],
               label='Observed', linestyle='--', alpha=0.7)
        
        # Mark anomalies
        if results['anomaly_scores']:
            sig_anomalies = [
                a for a in results['anomaly_scores']
                if a.signal == sig and a.is_anomaly
            ]
            if sig_anomalies:
                anomaly_times = [a.time for a in sig_anomalies]
                anomaly_values = [
                    df_obs[df_obs['time'] == t][sig].values[0]
                    for t in anomaly_times
                    if len(df_obs[df_obs['time'] == t]) > 0
                ]
                ax.scatter(anomaly_times, anomaly_values,
                          color='red', marker='x', s=100,
                          label='Anomaly', zorder=5)
        
        # Mark agent actions
        if results['agent_actions']:
            action_times = [a.time for a in results['agent_actions']]
            for at in action_times:
                ax.axvline(at, color='green', alpha=0.3, linestyle=':')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time [h]')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Use last subplot for summary text
    axes[-1].axis('off')
    summary_text = f"""
    RUN SUMMARY
    {'='*30}
    Run ID: {results['run_id']}
    
    Final Titer: {results['final_titer']:.2f} mg/mL
    Final Biomass: {results['final_biomass']:.2f} g/L
    
    Anomalies Detected: {results['num_anomalies']}
    Agent Actions: {results['num_actions']}
    """
    axes[-1].text(0.1, 0.5, summary_text, 
                 fontsize=11, family='monospace',
                 verticalalignment='center')
    
    plt.suptitle(f"BioPilot Run: {results['run_id']}", 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # For Databricks notebook execution
    # spark is already available in notebook context
    
    print("BioPilot Workflow - Complete Integration Demo")
    print("="*60)
    
    # Example 1: Single run with overfeed fault
    print("\n### Example 1: Single Run with Overfeed Fault ###")
    results, workflow = run_example_workflow(spark)
    
    # Visualize results
    visualize_run(results, save_path="/tmp/biopilot_run.png")
    
    # Example 2: Query results from data lake
    print("\n### Example 2: Query Data Lake ###")
    run_summary = workflow.data_lake.get_run_summary(spark, results['run_id'])
    print(f"\nTelemetry points: {run_summary['telemetry_points']}")
    print(f"Anomalies: {run_summary['anomalies_detected']}")
    print(f"Actions: {run_summary['actions_taken']}")
    
    # Example 3: Batch run multiple scenarios
    print("\n### Example 3: Batch Scenario Testing ###")
    scenarios = ['baseline', 'overfeed', 'underfeed', 'DO_drop']
    batch_summary = run_batch_scenarios(spark, scenarios, num_replicates=3)
    
    print("\nBatch run complete!")
    print(f"Total runs: {len(batch_summary)}")
    
    # Example 4: List all runs in data lake
    print("\n### Example 4: List All Runs ###")
    all_runs = workflow.data_lake.list_runs(spark, limit=20)
    print(all_runs[['run_id', 'scenario', 'final_titer', 'num_anomalies', 'success']])