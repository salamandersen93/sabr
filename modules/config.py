"""
config.py - ENHANCED

Enhanced configuration for BioPilot with expanded fault scenarios,
game mechanics, and scenario definitions.
"""

from typing import Dict

# Simulation control
SIMULATION_PARAMS: Dict = {
    "dt": 1,             # timestep [h]
    "total_time": 120,   # total simulation time [h]
    "random_seed": 42      # gaussian noise
}

# Initial state (true process)
INITIAL_STATE: Dict = {
    "X": 0.1,       # biomass [g/L]
    "S_glc": 20.0,  # glucose [g/L]
    "P": 0.0,       # product titer [mg/mL]
    "DO": 100.0,    # dissolved oxygen [% saturation]
    "pH": 7.20      # pH
}

# Kinetic parameters 
KINETIC_PARAMS: Dict = {
    # Monod parameters - growth rate
    "mu_max": 0.04,      # 1/h, plausible CHO demo value
    "Ks_glc": 0.5,       # g/L, Monod half-saturation for glucose
    "Y_xs": 0.5,         # g biomass / g glucose

    # Product formation: dP/dt = alpha*mu*X + beta*X
    "alpha": 0.01,       # mg product / (g biomass * h) scaling for growth-associated
    "beta": 0.0005,      # mg product / (g biomass * h) basal production

    # Death and maintenance
    "kd": 0.005,         # 1/h, death rate

    # Oxygen model
    "kLa": 10.0,         # 1/h, volumetric mass transfer coefficient (proxy)
    "o2_uptake_coeff": 0.02,  # % DO consumed per (g/L·h) by biomass, proxy for OUR

    # pH model
    "acid_prod_coeff": 1e-4,          # mol H+/g/h
    "acid_from_substrate": 1e-5,      # mol H+/mmol glucose
    "buffer_capacity": 0.025,    # mol/(L·pH)
    "base_dose_mol": 2e-4,       # mol per dose
    "pH_setpoint": 7.2,
    "pH_deadband": 0.05,

    # Temperature / agitation modifiers (multiplicative factors; 1.0 = no change)
    "temp_factor": 1.0,  # multiplies mu_max (can be set by temp shift)
    "agitation_factor": 1.0  # affects kLa (kLa * agitation_factor)
}

# Reactor / feed parameters
REACTOR_PARAMS: Dict = {
    "V0": 2.0,                  # L, starting volume
    "feed_start_h": 24.0,       # h, time when feed begins
    "feed_rate_g_L_h": 0.05,    # g glucose / L / h (continuous feed rate)
    "feed_glc_conc": 400.0      # g/L, concentrated feed stock composition
}

# Sensor / observation params
SENSOR_PARAMS: Dict = {
    "sensor_noise_sigma": 0.001,   # fraction or absolute noise ~ (units); tune per signal
    "sensor_drift_rate": 0.0005,  # linear drift per hour for sensors (applied to observed)
    "sensor_dropout_prob": 0.0    # probability sensor is frozen in a run (0-1)
}

# Assay / offline measurement params
ASSAY_PARAMS: Dict = {
    "sampling_interval_h": 24.0,  # h between offline assay samples
    "assay_lag_h": 4.0,           # h delay before assay result is available
    "assay_cv": 0.08,             # coefficient of variation (8%) for assay measurement noise
    "assay_LOD_titer": 0.001      # mg/mL, limit of detection for titer
}

# EXPANDED Fault templates - production scenarios
FAULT_TEMPLATES: Dict = {
    "overfeed": {
        "type": "overfeed",
        "description": "Glucose overfeed causing substrate accumulation",
        "start_h": 20.0,
        "duration_h": 2.0,
        "magnitude_multiplier": 1.5,
        "severity": 3
    },
    "underfeed": {
        "type": "underfeed",
        "description": "Feed pump failure causing substrate depletion",
        "start_h": 40.0,
        "duration_h": 2.0,
        "magnitude_multiplier": 0.5,
        "severity": 4
    },
    "DO_drop": {
        "type": "DO_drop",
        "description": "Aeration failure leading to oxygen depletion",
        "start_h": 60.0,
        "duration_h": 4.0,
        "magnitude_abs": -40.0,
        "severity": 5
    },
    "sensor_freeze": {
        "type": "sensor_freeze",
        "description": "DO sensor malfunction - frozen reading",
        "start_h": 50.0,
        "duration_h": 3.0,
        "sensor": "DO",
        "severity": 2
    },
    "contamination": {
        "type": "contamination",
        "description": "Bacterial contamination increasing death rate",
        "start_h": 72.0,
        "duration_h": 10.0,
        "death_rate_multiplier": 3.0,
        "severity": 5
    },
    "temp_shift_cold": {
        "type": "temp_shift",
        "description": "Cooling system overcorrection",
        "start_h": 30.0,
        "duration_h": 6.0,
        "temp_factor": 0.6,
        "severity": 3
    },
    "temp_shift_hot": {
        "type": "temp_shift",
        "description": "Temperature controller failure - overheating",
        "start_h": 30.0,
        "duration_h": 6.0,
        "temp_factor": 1.4,
        "severity": 4
    },
    "pH_drift": {
        "type": "pH_drift",
        "description": "Base pump failure causing acidification",
        "start_h": 45.0,
        "duration_h": 8.0,
        "acid_rate_multiplier": 3.0,
        "severity": 4
    },
    "agitation_failure": {
        "type": "agitation_failure",
        "description": "Impeller malfunction reducing mixing",
        "start_h": 55.0,
        "duration_h": 5.0,
        "agitation_multiplier": 0.3,
        "severity": 4
    },
    "feed_contamination": {
        "type": "feed_contamination",
        "description": "Contaminated feed causing immediate culture stress",
        "start_h": 35.0,
        "duration_h": 1.0,
        "death_spike": 0.05,
        "severity": 5
    },
    "standard": {
        "type": "standard", 
        "description": "no fault injected."}
}

# Game / scenario params
GAME_PARAMS: Dict = {
    "assay_budget": 6,
    "time_pressure": False,
    "severity_level_map": {1: 0.5, 2: 0.75, 3: 1.0, 4: 1.25, 5: 1.5},
    "target_titer": 8.0,  # mg/mL minimum for success
    "target_viability": 0.7  # minimum viability at harvest
}

# Scenario definitions for training/gameplay
SCENARIOS: Dict = {
    "tutorial_baseline": {
        "name": "Tutorial - Perfect Run",
        "description": "No faults, ideal conditions for learning",
        "faults": [],
        "difficulty": 1,
        "expected_outcome": "High titer, smooth kinetics"
    },
    "level_1_overfeed": {
        "name": "Level 1 - Feed Surge",
        "description": "Single overfeed event early in run",
        "faults": ["overfeed"],
        "difficulty": 2,
        "hints": ["Watch substrate levels", "Consider reducing feed rate"]
    },
    "level_2_sensor": {
        "name": "Level 2 - Sensor Malfunction",
        "description": "DO sensor freezes mid-run",
        "faults": ["sensor_freeze"],
        "difficulty": 2,
        "hints": ["Compare DO with expected trends", "Check other correlated signals"]
    },
    "level_3_do_crisis": {
        "name": "Level 3 - Oxygen Crisis",
        "description": "Critical DO drop requiring immediate intervention",
        "faults": ["DO_drop"],
        "difficulty": 4,
        "hints": ["Increase agitation immediately", "Monitor biomass response"]
    },
    "level_4_multi_fault": {
        "name": "Level 4 - Cascade Failure",
        "description": "Multiple faults requiring prioritization",
        "faults": ["underfeed", "sensor_freeze", "pH_drift"],
        "difficulty": 5,
        "hints": ["Prioritize life-threatening faults first", "Use assays strategically"]
    },
    "expert_contamination": {
        "name": "Expert - Contamination Event",
        "description": "Detect and respond to culture contamination",
        "faults": ["contamination"],
        "difficulty": 5,
        "hints": ["Death rate will increase", "Early detection is critical"]
    },
    "expert_thermal": {
        "name": "Expert - Temperature Excursion",
        "description": "Temperature control failure",
        "faults": ["temp_shift_hot"],
        "difficulty": 4,
        "hints": ["Growth rate will change", "Product quality may be affected"]
    }
}

# Scoring weights for game evaluation
SCORING_CONFIG: Dict = {
    "titer_weight": 0.4,
    "speed_weight": 0.2,
    "efficiency_weight": 0.2,  # actions per outcome
    "detection_weight": 0.2,    # anomaly detection accuracy
    "bonus_perfect": 50,        # points for perfect run
    "penalty_critical_fault": -20  # penalty for missing critical fault
}

# convenience export list
__all__ = [
    "SIMULATION_PARAMS", "INITIAL_STATE", "KINETIC_PARAMS",
    "REACTOR_PARAMS", "SENSOR_PARAMS", "ASSAY_PARAMS",
    "FAULT_TEMPLATES", "GAME_PARAMS", "SCENARIOS", "SCORING_CONFIG"
]