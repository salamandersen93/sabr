"""
anomaly_detection.py

Anomaly detection methods for bioreactor telemetry.
Includes classical statistical methods and ML-ready framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from collections import deque


@dataclass
class AnomalyScore:
    """Container for anomaly detection results."""
    time: float
    signal: str
    score: float
    is_anomaly: bool
    method: str
    context: Optional[Dict] = None


class MovingWindowDetector:
    """Detect anomalies using moving window statistics."""
    
    def __init__(self, window_size: int = 20, threshold_sigma: float = 3.0):
        self.window_size = window_size
        self.threshold_sigma = threshold_sigma
        self.windows = {}  # signal -> deque of recent values
    
    def detect(self, signal_name: str, value: float, time: float) -> AnomalyScore:
        """Detect if current value is anomalous based on recent history."""
        
        if signal_name not in self.windows:
            self.windows[signal_name] = deque(maxlen=self.window_size)
        
        window = self.windows[signal_name]
        
        # Need sufficient history
        if len(window) < self.window_size:
            window.append(value)
            return AnomalyScore(
                time=time,
                signal=signal_name,
                score=0.0,
                is_anomaly=False,
                method='moving_window',
                context={'window_full': False}
            )
        
        # Compute statistics on window
        mean = np.mean(window)
        std = np.std(window)
        
        # Z-score
        if std > 0:
            z_score = abs((value - mean) / std)
        else:
            z_score = 0.0
        
        is_anomaly = z_score > self.threshold_sigma
        
        # Add current value to window
        window.append(value)
        
        return AnomalyScore(
            time=time,
            signal=signal_name,
            score=z_score,
            is_anomaly=is_anomaly,
            method='moving_window',
            context={
                'mean': mean,
                'std': std,
                'threshold': self.threshold_sigma
            }
        )


class CUSUMDetector:
    """CUSUM (Cumulative Sum) control chart for detecting shifts."""
    
    def __init__(self, target_mean: float, std_dev: float, 
                 k: float = 0.5, h: float = 5.0):
        """
        Args:
            target_mean: Expected mean of the process
            std_dev: Expected standard deviation
            k: Allowable slack (typically 0.5 * std_dev)
            h: Decision interval (typically 4-5 * std_dev)
        """
        self.target = target_mean
        self.std = std_dev
        self.k = k * std_dev
        self.h = h * std_dev
        
        self.cusum_high = 0.0
        self.cusum_low = 0.0
    
    def detect(self, signal_name: str, value: float, time: float) -> AnomalyScore:
        """Update CUSUM and check for anomaly."""
        
        # Standardize
        deviation = value - self.target
        
        # Update CUSUM
        self.cusum_high = max(0, self.cusum_high + deviation - self.k)
        self.cusum_low = max(0, self.cusum_low - deviation - self.k)
        
        # Check thresholds
        is_anomaly = (self.cusum_high > self.h) or (self.cusum_low > self.h)
        score = max(self.cusum_high, self.cusum_low) / self.h
        
        return AnomalyScore(
            time=time,
            signal=signal_name,
            score=score,
            is_anomaly=is_anomaly,
            method='cusum',
            context={
                'cusum_high': self.cusum_high,
                'cusum_low': self.cusum_low,
                'threshold': self.h
            }
        )
    
    def reset(self):
        """Reset CUSUM counters."""
        self.cusum_high = 0.0
        self.cusum_low = 0.0


class RateOfChangeDetector:
    """Detect anomalies based on rate of change (derivative)."""
    
    def __init__(self, max_rate: float):
        """
        Args:
            max_rate: Maximum allowed rate of change per time unit
        """
        self.max_rate = max_rate
        self.prev_value = None
        self.prev_time = None
    
    def detect(self, signal_name: str, value: float, time: float) -> AnomalyScore:
        """Detect excessive rate of change."""
        
        if self.prev_value is None:
            self.prev_value = value
            self.prev_time = time
            return AnomalyScore(
                time=time,
                signal=signal_name,
                score=0.0,
                is_anomaly=False,
                method='rate_of_change',
                context={'initialized': True}
            )
        
        dt = time - self.prev_time
        if dt <= 0:
            return AnomalyScore(time=time, signal=signal_name, score=0.0, 
                              is_anomaly=False, method='rate_of_change')
        
        rate = abs(value - self.prev_value) / dt
        score = rate / self.max_rate
        is_anomaly = rate > self.max_rate
        
        self.prev_value = value
        self.prev_time = time
        
        return AnomalyScore(
            time=time,
            signal=signal_name,
            score=score,
            is_anomaly=is_anomaly,
            method='rate_of_change',
            context={'rate': rate, 'threshold': self.max_rate}
        )


class MultiSignalCorrelationDetector:
    """Detect anomalies based on expected correlations between signals."""
    
    def __init__(self, expected_correlations: Dict[Tuple[str, str], Tuple[float, float]]):
        """
        Args:
            expected_correlations: Dict mapping (signal1, signal2) -> (expected_corr, tolerance)
        """
        self.expected = expected_correlations
        self.history = {}  # signal -> list of recent values
        self.window_size = 50
    
    def detect(self, signals: Dict[str, float], time: float) -> List[AnomalyScore]:
        """Check if signal correlations are as expected."""
        
        results = []
        
        # Update history
        for sig, val in signals.items():
            if sig not in self.history:
                self.history[sig] = deque(maxlen=self.window_size)
            self.history[sig].append(val)
        
        # Check each expected correlation
        for (sig1, sig2), (expected_corr, tolerance) in self.expected.items():
            if sig1 in self.history and sig2 in self.history:
                if len(self.history[sig1]) >= 10 and len(self.history[sig2]) >= 10:
                    # Compute correlation
                    corr = np.corrcoef(
                        list(self.history[sig1]),
                        list(self.history[sig2])
                    )[0, 1]
                    
                    if np.isnan(corr):
                        corr = 0.0
                    
                    deviation = abs(corr - expected_corr)
                    is_anomaly = deviation > tolerance
                    
                    results.append(AnomalyScore(
                        time=time,
                        signal=f"{sig1}_{sig2}_corr",
                        score=deviation / tolerance if tolerance > 0 else 0.0,
                        is_anomaly=is_anomaly,
                        method='correlation',
                        context={
                            'observed_corr': corr,
                            'expected_corr': expected_corr,
                            'tolerance': tolerance
                        }
                    ))
        
        return results


class AnomalyDetectionEngine:
    """Composite anomaly detection system."""
    
    def __init__(self, config: Dict):
        """
        Initialize with configuration for each signal.
        
        config format:
        {
            'X': {
                'moving_window': {'window_size': 20, 'threshold_sigma': 3.0},
                'rate_of_change': {'max_rate': 0.5}
            },
            'DO': {...},
            'correlations': {
                ('X', 'DO'): {'expected': -0.7, 'tolerance': 0.3}
            }
        }
        """
        self.config = config
        self.detectors = {}
        
        # Initialize detectors for each signal
        for signal, methods in config.items():
            if signal == 'correlations':
                continue
                
            self.detectors[signal] = {}
            
            if 'moving_window' in methods:
                params = methods['moving_window']
                self.detectors[signal]['moving_window'] = MovingWindowDetector(**params)
            
            if 'rate_of_change' in methods:
                params = methods['rate_of_change']
                self.detectors[signal]['rate_of_change'] = RateOfChangeDetector(**params)
        
        # Correlation detector
        if 'correlations' in config:
            corr_config = {}
            for pair, params in config['correlations'].items():
                corr_config[pair] = (params['expected'], params['tolerance'])
            self.correlation_detector = MultiSignalCorrelationDetector(corr_config)
        else:
            self.correlation_detector = None
    
    def detect_step(self, signals: Dict[str, float], time: float) -> List[AnomalyScore]:
        """Run all detectors for current timestep."""
        
        results = []
        
        # Per-signal detectors
        for signal, value in signals.items():
            if signal in self.detectors:
                for method_name, detector in self.detectors[signal].items():
                    score = detector.detect(signal, value, time)
                    results.append(score)
        
        # Correlation detector
        if self.correlation_detector:
            corr_results = self.correlation_detector.detect(signals, time)
            results.extend(corr_results)
        
        return results
    
    def get_anomaly_summary(self, results: List[AnomalyScore]) -> Dict:
        """Summarize anomaly detection results."""
        
        total = len(results)
        anomalies = [r for r in results if r.is_anomaly]
        
        summary = {
            'total_checks': total,
            'anomalies_detected': len(anomalies),
            'anomaly_rate': len(anomalies) / total if total > 0 else 0.0,
            'by_signal': {},
            'by_method': {}
        }
        
        # Group by signal
        for result in anomalies:
            sig = result.signal
            if sig not in summary['by_signal']:
                summary['by_signal'][sig] = 0
            summary['by_signal'][sig] += 1
        
        # Group by method
        for result in anomalies:
            method = result.method
            if method not in summary['by_method']:
                summary['by_method'][method] = 0
            summary['by_method'][method] += 1
        
        return summary


# Helper: Create default detection config for bioreactor
def create_default_bioreactor_config() -> Dict:
    """Create sensible default anomaly detection config for CHO bioreactor."""
    
    return {
        'X': {
            'moving_window': {'window_size': 20, 'threshold_sigma': 3.0},
            'rate_of_change': {'max_rate': 0.1}  # g/L per hour
        },
        'S_glc': {
            'moving_window': {'window_size': 20, 'threshold_sigma': 3.0},
            'rate_of_change': {'max_rate': 2.0}  # g/L per hour
        },
        'DO': {
            'moving_window': {'window_size': 15, 'threshold_sigma': 2.5},
            'rate_of_change': {'max_rate': 15.0}  # % per hour
        },
        'pH': {
            'moving_window': {'window_size': 20, 'threshold_sigma': 2.5},
            'rate_of_change': {'max_rate': 0.1}  # pH units per hour
        },
        'P': {
            'moving_window': {'window_size': 30, 'threshold_sigma': 3.0},
            'rate_of_change': {'max_rate': 0.5}  # mg/mL per hour
        },
        'correlations': {
            ('X', 'DO'): {'expected': -0.6, 'tolerance': 0.4},
            ('X', 'S_glc'): {'expected': -0.5, 'tolerance': 0.4}
        }
    }