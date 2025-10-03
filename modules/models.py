"""
models.py

Deterministic process update functions for the synthetic bioreactor (MVP).
- Compute mu using Monod for glucose.
- Update biomass, substrate, product, DO, and pH deterministically
- Fault injection logic
- Models are kept simple, documented, and unit-consistent for use in the game engine.
"""
import sys
from pathlib import Path
import os

if "__file__" in globals():
    BASE_DIR = Path(__file__).resolve().parent
else:
    BASE_DIR = Path.cwd()
REPO_ROOT = BASE_DIR
for parent in BASE_DIR.parents:
    if (parent / "requirements.txt").exists() or (parent / ".git").exists():
        REPO_ROOT = parent
        break
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULES_DIR = REPO_ROOT / "app" / "modules"
if str(MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(MODULES_DIR))

from typing import Dict, Optional, List
from modules.sensor_noise import AdvancedSensorModel, apply_sensor_effects_enhanced
import numpy as np
import math

# helper function to clamp values between set bounds
def _clamp(x: float, lo: float = None, hi: float = None) -> float:
    if lo is not None and x < lo:
        return lo
    if hi is not None and x > hi:
        return hi
    return x

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

def update_biomass(X: float, mu: float, kinetics: Dict, dt: float) -> float:
    """
    dX/dt = (mu - kd) * X
    """
    kd = kinetics.get("kd", 0.005)
    dX = (mu * X - kd * X) * dt
    X_new = X + dX
    return _clamp(X_new, lo=0.0)

def update_substrate(X: float, S_glc: float, mu: float, 
                     feed_g_L_h: float, kinetics: Dict, dt: float) -> float:
    """
    Substrate (glucose) update function
    dS/dt = - (mu * X) / Y_xs + feed_rate (g/L/h)
    feed_g_L_h is the mass input normalized per L per hour.
    """
    Y_xs = kinetics.get("Y_xs", 0.5)
    dS = ( - (mu * X) / Y_xs + feed_g_L_h ) * dt
    S_new = S_glc + dS
    return _clamp(S_new, lo=0.0)

def update_product(X: float, P: float, mu: float, kinetics: Dict, dt: float) -> float:
    """
    Product titer update function

    dP/dt = alpha * mu * X + beta * X
    
    P is in g/L (not mg/mL - typical CHO titers are 1-10 g/L)
    
    Bioprocess basis:
    - alpha: specific productivity during growth (g product / g biomass)
    - beta: basal (non-growth-associated) specific productivity (g/(g·h))
    - Typical values: alpha ~ 0.5-2.0, beta ~ 0.01-0.05
    """
    # TODO: refine these default values based on literature
    alpha = kinetics.get("alpha", 1.0) 
    beta = kinetics.get("beta", 0.02)
    
    # Product formation: growth-associated + basal
    dP = (alpha * mu * X + beta * X) * dt
    P_new = P + dP
    return _clamp(P_new, lo=0.0)

def update_DO(X: float, DO: float, kinetics: Dict, dt: float) -> float:
    """
    Numerically stable DO update using the analytic solution for
    dDO/dt = kLa*(DO_sat - DO) - OUR
    """
    kLa = kinetics.get("kLa", 15.0)
    qO2 = kinetics.get("qO2", 5.0)  # ensure this is %DO/(g/L·h)
    if "o2_uptake_coeff" in kinetics:
        # Only keep this if you actually intend this conversion — review it.
        qO2 = kinetics["o2_uptake_coeff"] * 100

    DO_sat = 100.0
    OUR = qO2 * X  # %DO/h

    # Avoid division by zero on pathological kLa
    if kLa <= 0:
        # fallback to simple explicit update if kLa==0
        dDO = (-OUR) * dt
        return _clamp(DO + dDO, 0.0, 100.0)

    # steady-state DO for this instant
    DO_ss = DO_sat - OUR / kLa

    # analytic update (exact for linear ODE)
    DO_new = DO_ss + (DO - DO_ss) * math.exp(-kLa * dt)

    return _clamp(DO_new, lo=0.0, hi=100.0)

def update_pH(pH: float, X: float, kinetics: Dict, dt: float, 
              base_override: float = None) -> float:
    """
    pH update with acidification + base dosing.

    Acidification:
        dpH/dt = - acid_prod_coeff * X / buffer_capacity
    
    Base dosing (continuous pump version):
        If pH < setpoint - deadband:
            dpH = (effective_base_mol / buffer_capacity) * dt
    """
    # Acidification
    acid_coeff = kinetics.get("acid_prod_coeff", 1e-4)   # mol H+/g/h
    buffer_capacity = kinetics.get("buffer_capacity", 0.025)  # mol/(L·pH)
    dpH_acid = -(acid_coeff * X / buffer_capacity) * dt
    pH_new = pH + dpH_acid

    # Base dosing
    setpoint = kinetics.get("pH_setpoint", 7.0)
    deadband = kinetics.get("pH_deadband", 0.05)
    base_dose_mol = base_override if base_override is not None else kinetics.get("base_dose_mol", 2e-4)

    if pH_new < setpoint - deadband and base_dose_mol > 0:
        dpH_base = (base_dose_mol / buffer_capacity) * dt
        pH_new += dpH_base

    return _clamp(pH_new, lo=6.5, hi=7.6)

class FaultManager:
    """FAULT INJECTION SYSTEM.
    Manages fault injection during simulation runs."""
    
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
    
    def inject_faults(self, state: Dict, t: float, dt: float, kinetics: Dict, 
                      base_feed_rate: float) -> tuple[Dict, float, Dict]:
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
            
            # check status of fault (check if current time is within fault duration)
            if start_h <= t < (start_h + duration_h):
                
                # TODO: review all the logic and reasoning below, add as many faults as possible to this list
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
                    modified_state['_sensor_frozen'] = fault.get('sensor', 'DO')
                
                elif fault_type == 'contamination':
                    # Increase death rate
                    modified_kinetics['kd'] = kinetics['kd'] * 2.0
                
                elif fault_type == 'temp_shift':
                    temp_factor = fault.get('temp_factor', 0.7)
                    modified_kinetics['temp_factor'] = temp_factor

                elif fault_type == 'base_overdose':
                    multiplier = fault.get('magnitude_multiplier', 2)
                    modified_kinetics['base_dose_mol'] = kinetics['base_dose_mol'] * multiplier
                
                elif fault_type == 'base_underdose':
                    multiplier = fault.get('magnitude_multiplier', 0.5)
                    modified_kinetics['base_dose_mol'] = kinetics['base_dose_mol'] * multiplier 
                
                elif fault_type == 'base_pump_stuck':
                    multiplier = fault.get('magnitude_multiplier', 0)
                    modified_kinetics['base_dose_mol'] = kinetics['base_dose_mol'] * multiplier
                
                elif fault_type == 'sudden_base_bolus':
                    multiplier = fault.get('magnitude_multiplier', 10)
                    modified_kinetics['base_dose_mol'] = kinetics['base_dose_mol'] * multiplier
        
        return modified_state, modified_feed, modified_kinetics

# main class to run simulations
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
        
        # NEW: Initialize advanced sensor model
        self.sensor_model = AdvancedSensorModel()
        
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
            # apply faults
            state, feed_rate, kinetics = self.fault_manager.inject_faults(
                state, t, dt, self.kinetics, base_feed_rate
            )
            
            # get growth rate
            mu = compute_mu(state['S_glc'], kinetics)
            
            # update true state
            state['X'] = update_biomass(state['X'], mu, kinetics, dt)
            state['S_glc'] = update_substrate(state['X'], state['S_glc'], mu, feed_rate, kinetics, dt)
            state['P'] = update_product(state['X'], state['P'], mu, kinetics, dt)
            state['DO'] = update_DO(state['X'], state['DO'], kinetics, dt)
            
            # FIXED: pH update (was referencing undefined 'pH' variable and had wrong params)
            state['pH'] = update_pH(state['pH'], state['X'], kinetics, dt)
            
            true_snap = state.copy()
            true_snap['time'] = t
            true_snap['feed_rate'] = feed_rate
            true_history.append(true_snap)
            
            # UPDATED: Use enhanced sensor model
            frozen = [state.get('_sensor_frozen')] if '_sensor_frozen' in state else None
            obs_snap = apply_sensor_effects_enhanced(
                state, t, self.sensor_model, frozen,
                noise_multiplier=self.sensor_params.get('noise_multiplier', 1.0)
            )
            obs_snap['time'] = t
            observed_history.append(obs_snap)
        
        return true_history, observed_history
    
    def get_sensor_diagnostics(self) -> Dict[str, Dict]:
        """
        Get health status of all sensors.
        Useful for agent monitoring and fault detection.
        """
        sensors = ['X', 'S_glc', 'P', 'DO', 'pH']
        return {
            sensor: self.sensor_model.get_sensor_health(sensor)
            for sensor in sensors
        }
    
    def recalibrate_sensor(self, sensor: str):
        """
        Simulate sensor recalibration event.
        Resets drift and fouling for specified sensor.
        """
        self.sensor_model.reset_sensor(sensor)