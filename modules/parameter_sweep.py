import os
from datetime import datetime
import pandas as pd
from run_simulation_workflow import BioPilotWorkflow
from reporting import BioreactorPDFReport
from config import SIMULATION_PARAMS, INITIAL_STATE, KINETIC_PARAMS, REACTOR_PARAMS, SENSOR_PARAMS, FAULT_TEMPLATES

# setup
sweep_values = [0.03, 0.04, 0.05, 0.06]  # mu_max values
output_folder = f"/tmp/biopilot_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_folder, exist_ok=True)

reporter = BioreactorPDFReport(output_dir=output_folder)
sweep_results = []

# run sweep
for mu_max in sweep_values:
    print(f"\nRunning sweep for mu_max={mu_max}")
    config_sweep = {
        'SIMULATION_PARAMS': SIMULATION_PARAMS,
        'INITIAL_STATE': INITIAL_STATE,
        'KINETIC_PARAMS': {**KINETIC_PARAMS, 'mu_max': mu_max},
        'REACTOR_PARAMS': REACTOR_PARAMS,
        'SENSOR_PARAMS': SENSOR_PARAMS,
        'FAULT_TEMPLATES': FAULT_TEMPLATES
    }

    workflow_sweep = BioPilotWorkflow(spark=None, config_dict=config_sweep,
                                      enable_agent=True, enable_anomaly_detection=True)
    
    results = workflow_sweep.run_with_monitoring(base_feed_rate=0.1, save_to_lake=False)
    
    sweep_results.append({
        'mu_max': mu_max,
        'final_titer': results['final_titer'],
        'final_biomass': results['final_biomass'],
        'num_anomalies': results['num_anomalies'],
        'run_id': results['run_id']
    })
    
    # optionally save PDF per sweep run
    pdf_file = reporter.generate_summary_pdf(
        results=results,
        telemetry_df=results['observed_history'],
        ai_summary=results['agent_explain'],
        faults={},  # no faults for sweep
        param_config=config_sweep
    )
    pdf_path = os.path.join(output_folder, f"sweep_mu_{mu_max:.2f}_{results['run_id']}.pdf")
    os.rename(pdf_file, pdf_path)
    print(f"PDF report saved to {pdf_path}")

# save sweep summary CSV
sweep_df = pd.DataFrame(sweep_results)
csv_path = os.path.join(output_folder, "parameter_sweep_summary.csv")
sweep_df.to_csv(csv_path, index=False)
print(f"Sweep summary CSV saved to {csv_path}")