"""
Enhanced sensor noise and measurement characteristics for bioreactor simulation.

This module provides realistic sensor behavior modeling including:
- Variable noise types (Gaussian, proportional, quantization)
- Sensor-specific characteristics (accuracy, precision, range)
- Time-dependent effects (drift, fouling)
"""

import numpy as np
from typing import Dict, Optional, Literal
from dataclasses import dataclass

@dataclass
class SensorCharacteristics:
    """Physical characteristics of a sensor type."""
    # Base noise parameters
    absolute_noise_sigma: float  # Absolute noise level (in measurement units)
    proportional_noise_pct: float  # Proportional noise as % of signal
    
    # Measurement characteristics
    resolution: float  # Quantization step size
    min_range: float  # Minimum measurable value
    max_range: float  # Maximum measurable value
    
    # Time-dependent effects
    drift_rate: float  # Units per hour
    fouling_rate: float  # Signal degradation rate (for biomass, product sensors)
    
    # Response characteristics
    response_time: float  # Time constant for sensor lag (hours)
    
    # Noise type weighting
    gaussian_weight: float = 0.7  # Weight for white noise
    proportional_weight: float = 0.3  # Weight for signal-dependent noise

# Realistic sensor profiles based on bioprocess instrumentation
SENSOR_PROFILES = {
    'X': SensorCharacteristics(
        absolute_noise_sigma=0.05,  # g/L - capacitance probes are fairly noisy
        proportional_noise_pct=3.0,  # 3% of reading
        resolution=0.01,
        min_range=0.0,
        max_range=50.0,
        drift_rate=0.002,  # Slow upward drift from fouling
        fouling_rate=0.0001,  # Cell adhesion to probe
        response_time=0.05  # ~3 min lag
    ),
    'S_glc': SensorCharacteristics(
        absolute_noise_sigma=0.02,  # g/L - enzymatic sensors
        proportional_noise_pct=1.5,
        resolution=0.05,  # Coarser quantization
        min_range=0.0,
        max_range=100.0,
        drift_rate=0.01,  # Enzyme degradation causes drift
        fouling_rate=0.00005,
        response_time=0.1  # ~6 min response time
    ),
    'P': SensorCharacteristics(
        absolute_noise_sigma=0.03,  # g/L - protein/mAb titer
        proportional_noise_pct=2.0,
        resolution=0.02,
        min_range=0.0,
        max_range=15.0,
        drift_rate=0.005,
        fouling_rate=0.0002,  # Protein adhesion
        response_time=0.083  # ~5 min
    ),
    'DO': SensorCharacteristics(
        absolute_noise_sigma=0.3,  # %DO - polarographic/optical
        proportional_noise_pct=0.5,  # Very stable measurement
        resolution=0.1,
        min_range=0.0,
        max_range=100.0,
        drift_rate=-0.02,  # Slight downward drift (membrane degradation)
        fouling_rate=0.00001,
        response_time=0.017  # ~1 min - fastest response
    ),
    'pH': SensorCharacteristics(
        absolute_noise_sigma=0.01,  # pH units - glass electrode
        proportional_noise_pct=0.2,  # Extremely stable
        resolution=0.01,
        min_range=2.0,
        max_range=12.0,
        drift_rate=0.001,  # Minimal drift
        fouling_rate=0.00002,
        response_time=0.033  # ~2 min
    )
}

class AdvancedSensorModel:
    """
    Sophisticated sensor modeling with multiple noise sources and effects.
    """
    
    def __init__(self, sensor_profiles: Dict[str, SensorCharacteristics] = None):
        """
        Initialize sensor model.
        
        Args:
            sensor_profiles: Custom sensor characteristics, defaults to SENSOR_PROFILES
        """
        self.profiles = sensor_profiles or SENSOR_PROFILES
        self.sensor_states = {}  # Track internal sensor states (for lag, fouling)
        
    def add_sensor_noise(self, 
                        sensor: str, 
                        true_value: float, 
                        t: float,
                        noise_multiplier: float = 1.0) -> float:
        """
        Apply realistic noise to a sensor measurement.
        
        Args:
            sensor: Sensor name (X, S_glc, P, DO, pH)
            true_value: True process value
            t: Current simulation time (hours)
            noise_multiplier: Global noise scaling factor (for easy tuning)
            
        Returns:
            Observed value with noise and effects applied
        """
        if sensor not in self.profiles:
            # Fallback for unknown sensors
            return true_value + np.random.normal(0, 0.001)
        
        profile = self.profiles[sensor]
        
        # Initialize sensor state if needed
        if sensor not in self.sensor_states:
            self.sensor_states[sensor] = {
                'last_output': true_value,
                'cumulative_fouling': 0.0,
                'drift_offset': 0.0
            }
        
        state = self.sensor_states[sensor]
        
        # 1. GAUSSIAN (WHITE) NOISE
        gaussian_noise = np.random.normal(0, profile.absolute_noise_sigma * noise_multiplier)
        gaussian_component = profile.gaussian_weight * gaussian_noise
        
        # 2. PROPORTIONAL (SIGNAL-DEPENDENT) NOISE
        prop_noise = np.random.normal(0, abs(true_value) * profile.proportional_noise_pct / 100.0)
        proportional_component = profile.proportional_weight * prop_noise * noise_multiplier
        
        # 3. QUANTIZATION (ADC resolution)
        # Apply noise first, then quantize
        noisy_value = true_value + gaussian_component + proportional_component
        quantized_value = np.round(noisy_value / profile.resolution) * profile.resolution
        
        # 4. SENSOR LAG (first-order response)
        # Exponential moving average to simulate sensor time constant
        alpha = 1.0 - np.exp(-1.0 / (profile.response_time * 100))  # Assuming dt ~ 0.01h
        lagged_value = alpha * quantized_value + (1 - alpha) * state['last_output']
        
        # 5. DRIFT (time-dependent offset)
        state['drift_offset'] += profile.drift_rate * 0.01  # Assuming dt ~ 0.01h
        
        # 6. FOULING (multiplicative degradation for some sensors)
        # More relevant for biomass/product sensors with cell/protein adhesion
        state['cumulative_fouling'] += profile.fouling_rate * t * 0.01
        fouling_factor = 1.0 + state['cumulative_fouling']
        
        # Apply drift and fouling
        final_value = lagged_value * fouling_factor + state['drift_offset']
        
        # 7. RANGE LIMITING (saturation)
        final_value = np.clip(final_value, profile.min_range, profile.max_range)
        
        # Update state
        state['last_output'] = final_value
        
        return final_value
    
    def reset_sensor(self, sensor: str):
        """Reset a sensor's internal state (simulates recalibration)."""
        if sensor in self.sensor_states:
            self.sensor_states[sensor] = {
                'last_output': 0.0,
                'cumulative_fouling': 0.0,
                'drift_offset': 0.0
            }
    
    def get_sensor_health(self, sensor: str) -> Dict[str, float]:
        """
        Get current sensor health metrics.
        
        Returns dict with drift_offset, fouling_factor, etc.
        Useful for agent diagnostics.
        """
        if sensor not in self.sensor_states:
            return {'status': 'uninitialized'}
        
        state = self.sensor_states[sensor]
        return {
            'drift_offset': state['drift_offset'],
            'fouling_factor': 1.0 + state['cumulative_fouling'],
            'last_output': state['last_output']
        }


def apply_sensor_effects_enhanced(state: Dict, 
                                 t: float, 
                                 sensor_model: AdvancedSensorModel,
                                 frozen_sensors: Optional[list] = None,
                                 noise_multiplier: float = 1.0) -> Dict:
    """
    Enhanced replacement for apply_sensor_effects() in models.py.
    
    Args:
        state: True state dictionary
        t: Current time (hours)
        sensor_model: Configured AdvancedSensorModel instance
        frozen_sensors: List of sensors currently frozen (faults)
        noise_multiplier: Global noise scaling (1.0 = normal, >1 = more noise)
    
    Returns:
        Observed state with realistic sensor effects
    """
    obs_state = state.copy()
    frozen = frozen_sensors or []
    
    sensors = ['X', 'S_glc', 'P', 'DO', 'pH']
    
    for sensor in sensors:
        if sensor in state and sensor not in frozen:
            obs_state[sensor] = sensor_model.add_sensor_noise(
                sensor=sensor,
                true_value=state[sensor],
                t=t,
                noise_multiplier=noise_multiplier
            )
    
    return obs_state