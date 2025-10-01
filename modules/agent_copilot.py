"""
agent_copilot.py

Multi-agent copilot system for bioreactor management.
Observes telemetry, flags anomalies, suggests interventions, and generates reports.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
import mlflow.deployments
import json


class ActionType(Enum):
    """Types of control actions the agent can take."""
    ADJUST_FEED = "adjust_feed"
    REQUEST_ASSAY = "request_assay"
    ADJUST_TEMPERATURE = "adjust_temperature"
    ADJUST_AGITATION = "adjust_agitation"
    ADJUST_PH = "adjust_ph"
    ALERT = "alert"
    DO_NOTHING = "do_nothing"


@dataclass
class AgentAction:
    """Represents an action proposed or taken by the agent."""
    action_type: ActionType
    parameters: Dict
    confidence: float  # 0-1
    rationale: str
    time: float
    priority: int = 1  # 1=low, 3=medium, 5=high
    
    def to_dict(self) -> Dict:
        return {
            'action_type': self.action_type.value,
            'parameters': self.parameters,
            'confidence': self.confidence,
            'rationale': self.rationale,
            'time': self.time,
            'priority': self.priority
        }


@dataclass
class AgentObservation:
    """What the agent sees at each timestep."""
    time: float
    telemetry: Dict[str, float]  # Signal name -> value
    recent_anomalies: List  # Recent AnomalyScore objects
    recent_actions: List[AgentAction]
    available_budget: Dict  # Resources available (assays, etc.)

# ----------------------------------------------------------------------
# Agent implementations
# ----------------------------------------------------------------------

class ExplainerAgent:
    def __init__(self, llama_client: LlamaCrewAgent):
        self.client = llama_client

    def explain(self, telemetry_snapshot: dict, anomalies: list, actions: list) -> str:
        prompt = f"""
        You are a bioprocess expert. Analyze the following CHO cell
        bioreactor conditions and provide possible explanations.

        Telemetry: {telemetry_snapshot}
        Anomalies: {anomalies}
        Actions: {actions}

        Give a concise, mechanistic explanation of why these
        conditions might arise in a fed-batch CHO culture.
        """
        response = self.client(prompt, temperature=0.3, max_tokens=300)
        return response


class RuleBasedAgent:
    """Rule-based control agent for bioreactor management."""
    
    def __init__(self, rules_config: Dict):
        self.rules = rules_config
        self.action_history = []
    
    def observe_and_act(self, observation: AgentObservation) -> List[AgentAction]:
        actions = []
        telemetry = observation.telemetry
        time = observation.time
        
        # Rule 1: Substrate management
        if 'S_glc' in telemetry:
            S = telemetry['S_glc']
            feed_rules = self.rules.get('feed_adjustment', {})
            
            if S < feed_rules.get('substrate_low_threshold', 2.0):
                actions.append(AgentAction(
                    action_type=ActionType.ADJUST_FEED,
                    parameters={'multiplier': feed_rules.get('feed_increase_factor', 1.2)},
                    confidence=0.8,
                    rationale=f"Substrate low ({S:.2f} g/L), increasing feed rate",
                    time=time,
                    priority=4
                ))
            
            elif S > feed_rules.get('substrate_high_threshold', 40.0):
                actions.append(AgentAction(
                    action_type=ActionType.ADJUST_FEED,
                    parameters={'multiplier': feed_rules.get('feed_decrease_factor', 0.8)},
                    confidence=0.7,
                    rationale=f"Substrate high ({S:.2f} g/L), reducing feed rate",
                    time=time,
                    priority=3
                ))
        
        # Rule 2: DO control
        if 'DO' in telemetry:
            DO = telemetry['DO']
            do_rules = self.rules.get('DO_control', {})
            
            if DO < do_rules.get('low_threshold', 30.0):
                actions.append(AgentAction(
                    action_type=ActionType.ADJUST_AGITATION,
                    parameters={'multiplier': do_rules.get('agitation_increase_factor', 1.1)},
                    confidence=0.9,
                    rationale=f"DO critically low ({DO:.1f}%), increasing agitation",
                    time=time,
                    priority=5
                ))
        
        # Rule 3: pH control
        if 'pH' in telemetry:
            pH = telemetry['pH']
            ph_rules = self.rules.get('pH_control', {})
            
            if pH < ph_rules.get('low_threshold', 6.8):
                actions.append(AgentAction(
                    action_type=ActionType.ADJUST_PH,
                    parameters={'base_addition': 0.1},
                    confidence=0.85,
                    rationale=f"pH low ({pH:.2f}), adding base",
                    time=time,
                    priority=4
                ))
            
            elif pH > ph_rules.get('high_threshold', 7.4):
                actions.append(AgentAction(
                    action_type=ActionType.ADJUST_PH,
                    parameters={'acid_addition': 0.1},
                    confidence=0.85,
                    rationale=f"pH high ({pH:.2f}), adding acid",
                    time=time,
                    priority=4
                ))
        
        # Rule 4: Anomaly response
        if observation.recent_anomalies:
            high_priority_anomalies = [
                a for a in observation.recent_anomalies 
                if a.is_anomaly and a.score > 5.0
            ]
            
            if high_priority_anomalies:
                if observation.available_budget.get('assays', 0) > 0:
                    actions.append(AgentAction(
                        action_type=ActionType.REQUEST_ASSAY,
                        parameters={'assay_type': 'titer'},
                        confidence=0.7,
                        rationale=f"High anomaly detected, requesting assay",
                        time=time,
                        priority=3
                    ))
        
        # Rule 5: Routine monitoring
        assay_interval = self.rules.get('assay_schedule', {}).get('interval_h', 24.0)
        if len(self.action_history) == 0 or \
           (time - self._last_assay_time() > assay_interval):
            if observation.available_budget.get('assays', 0) > 0:
                actions.append(AgentAction(
                    action_type=ActionType.REQUEST_ASSAY,
                    parameters={'assay_type': 'titer'},
                    confidence=0.6,
                    rationale="Routine scheduled assay",
                    time=time,
                    priority=2
                ))
        
        self.action_history.extend(actions)
        return actions
    
    def _last_assay_time(self) -> float:
        assay_actions = [
            a for a in self.action_history 
            if a.action_type == ActionType.REQUEST_ASSAY
        ]
        if assay_actions:
            return assay_actions[-1].time
        return 0.0


class PredictiveAgent:
    """TODO: ML-based predictive agent (placeholder for future ML models)."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
    
    def predict_trajectory(self, current_state: Dict, 
                          horizon_hours: float = 24.0) -> Dict:
        X = current_state.get('X', 0.1)
        P = current_state.get('P', 0.0)
        growth_rate = 0.03
        predicted_X = X * np.exp(growth_rate * horizon_hours)
        predicted_P = P + 0.01 * predicted_X * horizon_hours
        
        return {
            'predicted_titer': predicted_P,
            'predicted_biomass': predicted_X,
            'confidence': 0.5,
            'risk_factors': []
        }
    
    def recommend_action(self, prediction: Dict, 
                        current_state: Dict) -> Optional[AgentAction]:
        if prediction['predicted_titer'] < 5.0:
            return AgentAction(
                action_type=ActionType.ADJUST_FEED,
                parameters={'multiplier': 1.1},
                confidence=prediction['confidence'],
                rationale="Predicted low titer, increasing feed",
                time=current_state.get('time', 0.0),
                priority=3
            )
        return None


class MultiAgentCopilot:
    """Orchestrates multiple agent types and generates recommendations."""
    
    def __init__(self, rule_agent_config: Dict, 
                 use_predictive: bool = False):
        self.rule_agent = RuleBasedAgent(rule_agent_config)
        self.predictive_agent = PredictiveAgent() if use_predictive else None
        
        self.all_actions = []
        self.observation_history = []
    
    def process_observation(self, observation: AgentObservation) -> List[AgentAction]:
        self.observation_history.append(observation)
        
        actions = []
        rule_actions = self.rule_agent.observe_and_act(observation)
        actions.extend(rule_actions)
        
        if self.predictive_agent:
            prediction = self.predictive_agent.predict_trajectory(
                observation.telemetry
            )
            pred_action = self.predictive_agent.recommend_action(
                prediction, 
                observation.telemetry
            )
            if pred_action:
                actions.append(pred_action)
        
        actions.sort(key=lambda a: a.priority, reverse=True)
        self.all_actions.extend(actions)
        
        return actions
    
    def generate_report(self, time_window: Optional[Tuple[float, float]] = None) -> str:
        if time_window:
            start, end = time_window
            actions = [a for a in self.all_actions if start <= a.time <= end]
            observations = [o for o in self.observation_history 
                          if start <= o.time <= end]
        else:
            actions = self.all_actions
            observations = self.observation_history
        
        report = []
        report.append("=" * 60)
        report.append("AGENT COPILOT SUMMARY REPORT")
        report.append("=" * 60)
        report.append(f"Total Actions Proposed: {len(actions)}")
        
        action_types = {}
        for action in actions:
            at = action.action_type.value
            action_types[at] = action_types.get(at, 0) + 1
        
        report.append("Actions by Type:")
        for at, count in sorted(action_types.items(), 
                               key=lambda x: x[1], reverse=True):
            report.append(f"  {at}: {count}")

        high_priority = [a for a in actions if a.priority >= 4]
        if high_priority:
            report.append(f"High Priority Actions ({len(high_priority)}):")
            for action in high_priority[:10]:
                report.append(f"  t={action.time:.1f}h: {action.rationale}")
            report.append("")
        
        total_anomalies = sum(
            len(obs.recent_anomalies) 
            for obs in observations
        )
        report.append(f"Total Anomalies Observed: {total_anomalies}")
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def get_action_dataframe(self) -> pd.DataFrame:
        records = [a.to_dict() for a in self.all_actions]
        return pd.DataFrame(records)

# ----------------------------------------------------------------------
# Default configuration factory
# ----------------------------------------------------------------------

def create_default_copilot_config() -> Dict:
    return {
        'feed_adjustment': {
            'substrate_low_threshold': 2.0,
            'substrate_high_threshold': 40.0,
            'feed_increase_factor': 1.2,
            'feed_decrease_factor': 0.8
        },
        'DO_control': {
            'low_threshold': 30.0,
            'critical_threshold': 20.0,
            'agitation_increase_factor': 1.15
        },
        'pH_control': {
            'low_threshold': 6.8,
            'high_threshold': 7.4,
            'base_addition_rate': 0.1,
            'acid_addition_rate': 0.1
        },
        'assay_schedule': {
            'interval_h': 24.0,
            'anomaly_triggered': True
        },
        'alert_thresholds': {
            'biomass_crash_rate': -0.1,
            'DO_critical': 15.0,
            'pH_critical_low': 6.5,
            'pH_critical_high': 7.6
        }
    }