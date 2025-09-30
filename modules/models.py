"""
models.py

Deterministic process update functions for the synthetic bioreactor (MVP).
- Compute mu using Monod for glucose.
- Update biomass, substrate, product, DO, and pH deterministically.
- Fault injection logic
- Models are kept simple, documented, and unit-consistent for use in the game engine.
"""

from typing import Dict, Optional, List
import numpy as np
import math

# Utility: clamp helper
def _clamp(x: float, lo: float = None, hi: float = None) -> float:
    if lo is not None and x < lo:
        return lo
    if hi is not None and x > hi:
        return hi
    return x

# Compute specific growth rate (Monod)
def compute_mu(S_glc: float, kinetics: Dict) -> float:
    """
    Compute specific growth rate using Monod model with glucose.
    mu = mu_max * (S / (Ks + S)) * temp_factor * agitation_factor
    """
    mu_max = kinetics.get("mu_max", 0.04)
    Ks = kinetics.get("Ks_glc", 0.5)
    temp_factor = kinetics.get("temp_factor", 1.0)
    agitation_factor = kinetics.get("agitation_factor", 1.0)

    # Avoid division by zero
    mu = mu_max * (S_glc / (Ks + S_glc)) if (Ks + S_glc) > 0 else 0.0
    mu *= temp_factor * agitation_factor
    return max(mu, 0.0)

# Biomass update
def update_biomass(X: float, mu: float, kinetics: Dict, dt: float) -> float:
    """
    dX/dt = (mu - kd) * X
    """
    kd = kinetics.get("kd", 0.005)
    dX = (mu * X - kd * X) * dt
    X_new = X + dX
    return _clamp(X_new, lo=0.0)

# Substrate (glucose) update
def update_substrate(X: float, S_glc: float, mu: float, feed_g_L_h: float, kinetics: Dict, dt: float) -> float:
    """
    dS/dt = - (mu * X) / Y_xs + feed_rate (g/L/h)
    feed_g_L_h is the mass input normalized per L per hour.
    """
    Y_xs = kinetics.get("Y_xs", 0.5)
    dS = ( - (mu * X) / Y_xs + feed_g_L_h ) * dt
    S_new = S_glc + dS
    return _clamp(S_new, lo=0.0)

# Product (titer) update
def update_product(X: float, P: float, mu: float, kinetics: Dict, dt: float) -> float:
    """
    dP/dt = alpha * mu * X + beta * X
    
    P is in g/L (not mg/mL - typical CHO titers are 1-10 g/L)
    
    Bioprocess basis:
    - alpha: specific productivity during growth (g product / g biomass)
    - beta: basal (non-growth-associated) specific productivity (g/(g·h))
    - Typical values: alpha ~ 0.5-2.0, beta ~ 0.01-0.05
    """
    alpha = kinetics.get("alpha", 1.0)  # Increased from 0.01
    beta = kinetics.get("beta", 0.02)   # Increased from 0.0005
    
    # Product formation: growth-associated + basal
    dP = (alpha * mu * X + beta * X) * dt
    P_new = P + dP
    return _clamp(P_new, lo=0.0)

# Dissolved oxygen update (proxy)
def update_DO(X: float, DO: float, kinetics: Dict, dt: float) -> float:
    """
    Simplified DO dynamic:
      dDO/dt = kLa*(DO_sat - DO) - OUR
    
    Bioprocess basis:
    - kLa: volumetric mass transfer coefficient (1/h)
      Typical stirred tank: 5-50 h⁻¹ depending on agitation/sparging
    - OUR: oxygen uptake rate (%DO/h)
      Approximated as qO2 * X where qO2 is specific oxygen uptake rate
      For CHO: qO2 ~ 0.05-0.2 mmol/(g·h) → roughly 2-8 %DO/(g/L·h)
    
    The driving force (DO_sat - DO) must balance OUR at steady state:
      kLa * (100 - DO_ss) = qO2 * X
      For X=3 g/L, qO2=5, kLa=15 → DO_ss = 100 - (5*3)/15 = 99% (good)
      For X=3 g/L, qO2=5, kLa=3  → DO_ss = 100 - (5*3)/3  = 95% (realistic drop)
    """
    kLa = kinetics.get("kLa", 15.0)  # Increased from 10.0 for better control
    qO2 = kinetics.get("qO2", 5.0)    # NEW: specific O2 uptake rate (%DO/(g/L·h))
    
    # For backward compatibility, support old parameter
    if "o2_uptake_coeff" in kinetics:
        qO2 = kinetics["o2_uptake_coeff"] * 100  # Convert if old param used
    
    # DO saturation is 100% in our normalized units
    DO_sat = 100.0
    
    # Oxygen uptake rate (OUR) in %DO/h
    OUR = qO2 * X
    
    # Mass balance: oxygen transfer in - oxygen consumed
    dDO = (kLa * (DO_sat - DO) - OUR) * dt
    DO_new = DO + dDO
    
    # Physical constraint: DO must be between 0-100%
    return _clamp(DO_new, lo=0.0, hi=100.0)

# pH update (slow deterministic drift)
def update_pH(pH: float, X: float, kinetics: Dict, dt: float) -> float:
    """
    Simple metabolic acidification:
      dpH/dt = - acid_rate * X
    acid_rate in pH units per (g/L·h)
    """
    acid_rate = kinetics.get("acid_rate", 0.0005)
    dpH = - acid_rate * X * dt
    pH_new = pH + dpH
    # Clamp to a plausible range for CHO culture
    return _clamp(pH_new, lo=6.5, hi=7.6)

# Gaussian noise addition
def add_noise(signal: float, sigma: float = 0.001) -> float:
    """Add Gaussian noise to a signal (for observed values)."""
    return signal + np.random.normal(0, sigma)

# ============================================
# FAULT INJECTION SYSTEM
# ============================================

class FaultManager:
    """Manages fault injection during simulation runs."""
    
    def __init__(self, fault_templates: Dict):
        self.fault_templates = fault_templates
        self.active_faults: List[Dict] = []
    
    def activate_fault(self, fault_name: str, start_time: float):
        """Activate a fault from templates at specified time."""
        if fault_name in self.fault_templates:
            fault = self.fault_templates[fault_name].copy()
            fault['start_h'] = start_time
            fault['name'] = fault_name
            self.active_faults.append(fault)
    
    def inject_faults(self, state: Dict, t: float, dt: float, 
                     kinetics: Dict, base_feed_rate: float) -> tuple[Dict, float, Dict]:
        """
        Apply active faults to current state.
        Returns: (modified_state, modified_feed_rate, modified_kinetics)
        """
        modified_state = state.copy()
        modified_feed = base_feed_rate
        modified_kinetics = kinetics.copy()
        
        for fault in self.active_faults:
            fault_type = fault.get('type')
            start_h = fault.get('start_h', 0)
            duration_h = fault.get('duration_h', 0)
            
            # Check if fault is active at current time
            if start_h <= t < (start_h + duration_h):
                
                if fault_type == 'overfeed':
                    multiplier = fault.get('magnitude_multiplier', 1.5)
                    modified_feed = base_feed_rate * multiplier
                
                elif fault_type == 'underfeed':
                    multiplier = fault.get('magnitude_multiplier', 0.5)
                    modified_feed = base_feed_rate * multiplier
                
                elif fault_type == 'DO_drop':
                    magnitude = fault.get('magnitude_abs', -40.0)
                    # Reduce kLa to simulate poor aeration
                    modified_kinetics['kLa'] = kinetics['kLa'] * 0.3
                
                elif fault_type == 'sensor_freeze':
                    # This would be handled at observation layer
                    # Mark in state metadata for observation handling
                    modified_state['_sensor_frozen'] = fault.get('sensor', 'DO')
                
                elif fault_type == 'contamination':
                    # Increase death rate
                    modified_kinetics['kd'] = kinetics['kd'] * 2.0
                
                elif fault_type == 'temp_shift':
                    temp_factor = fault.get('temp_factor', 0.7)
                    modified_kinetics['temp_factor'] = temp_factor
        
        return modified_state, modified_feed, modified_kinetics


def apply_sensor_effects(state: Dict, t: float, sensor_params: Dict, 
                        frozen_sensors: Optional[List[str]] = None) -> Dict:
    """
    Apply sensor noise, drift, and dropout to observed state.
    
    Args:
        state: True state dictionary
        t: Current time
        sensor_params: Sensor configuration
        frozen_sensors: List of sensors currently frozen
    
    Returns:
        Observed state with sensor effects applied
    """
    obs_state = state.copy()
    
    sigma = sensor_params.get('sensor_noise_sigma', 0.001)
    drift_rate = sensor_params.get('sensor_drift_rate', 0.0005)
    
    sensors = ['X', 'S_glc', 'P', 'DO', 'pH']
    frozen = frozen_sensors or []
    
    for sensor in sensors:
        if sensor in state:
            if sensor in frozen:
                # Sensor frozen - don't update (would need previous value stored)
                continue
            else:
                # Apply noise
                obs_state[sensor] = add_noise(state[sensor], sigma)
                # Apply drift
                obs_state[sensor] += drift_rate * t
    
    return obs_state


# ============================================
# BATCH SIMULATION RUNNER
# ============================================

class BioreactorSimulation:
    """Complete simulation runner with fault injection."""
    
    def __init__(self, config_dict: Dict):
        """
        Initialize from config dictionary containing all parameter groups.
        """
        self.sim_params = config_dict['SIMULATION_PARAMS']
        self.initial_state = config_dict['INITIAL_STATE']
        self.kinetics = config_dict['KINETIC_PARAMS']
        self.reactor_params = config_dict['REACTOR_PARAMS']
        self.sensor_params = config_dict['SENSOR_PARAMS']
        self.fault_templates = config_dict['FAULT_TEMPLATES']
        
        self.fault_manager = FaultManager(self.fault_templates)
        
        # Set random seed
        np.random.seed(self.sim_params['random_seed'])
    
    def run(self, base_feed_rate: float = 0.1) -> tuple[List[Dict], List[Dict]]:
        """
        Execute simulation.
        
        Returns:
            (true_history, observed_history) as lists of state dictionaries
        """
        dt = self.sim_params['dt']
        total_time = self.sim_params['total_time']
        time_points = np.arange(0, total_time, dt)
        
        true_history = []
        observed_history = []
        
        state = self.initial_state.copy()
        
        for t in time_points:
            # Apply faults
            state, feed_rate, kinetics = self.fault_manager.inject_faults(
                state, t, dt, self.kinetics, base_feed_rate
            )
            
            # Compute growth rate
            mu = compute_mu(state['S_glc'], kinetics)
            
            # Update true state
            state['X'] = update_biomass(state['X'], mu, kinetics, dt)
            state['S_glc'] = update_substrate(
                state['X'], state['S_glc'], mu, feed_rate, kinetics, dt
            )
            state['P'] = update_product(state['X'], state['P'], mu, kinetics, dt)
            state['DO'] = update_DO(state['X'], state['DO'], kinetics, dt)
            state['pH'] = update_pH(state['pH'], state['X'], kinetics, dt)
            
            # Record true state
            true_snap = state.copy()
            true_snap['time'] = t
            true_snap['feed_rate'] = feed_rate
            true_history.append(true_snap)
            
            # Create observed state with sensor effects
            frozen = [state.get('_sensor_frozen')] if '_sensor_frozen' in state else None
            obs_snap = apply_sensor_effects(state, t, self.sensor_params, frozen)
            obs_snap['time'] = t
            observed_history.append(obs_snap)
        
        return true_history, observed_history