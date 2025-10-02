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
import mlflow
from crewai import Agent, Task, Crew
import mlflow.deployments

class LlamaCrewAgent:
    def __init__(self, endpoint="databricks-meta-llama-3-3-70b-instruct"):
        self.client = mlflow.deployments.get_deploy_client("databricks")
        self.endpoint = endpoint

    def __call__(self, prompt, temperature=0.1, max_tokens=256):
        messages = [{"role": "user", "content": prompt}]
        response = self.client.predict(
            endpoint=self.endpoint,
            inputs={
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        )
        return response

# AGENTIC PROCESSES

class ExplainerAgent:
    def __init__(self, endpoint="databricks-meta-llama-3-3-70b-instruct"):
        self.client = mlflow.deployments.get_deploy_client("databricks")
        self.endpoint = endpoint

    def _serialize_df(self, df):
        """Convert DataFrame to compact JSON string."""
        if isinstance(df, pd.DataFrame):
            return df.to_json(orient="records")
        return str(df)

    def explain(self, telemetry_snapshot, anomalies):
        if isinstance(telemetry_snapshot, dict):
            telemetry_serialized = {k: self._serialize_df(v) for k, v in telemetry_snapshot.items()}
        elif isinstance(telemetry_snapshot, list):
            telemetry_serialized = [self._serialize_df(v) for v in telemetry_snapshot]
        else:
            raise TypeError("telemetry_snapshot must be a dict or list")
        anomalies_serialized = [self._serialize_df(a) for a in anomalies]

        llama_agent = Agent(
            role="Pharmaceutical Large Molecule Bioreactor Troubleshooting Expert",
            goal="Give a concise, mechanistic explanation of why given conditions and issues might arise in a fed-batch CHO culture.",
            backstory="You are a bioprocess expert. Analyze the following CHO cellbioreactor conditions and provide possible explanations. Categorize the primary root causes of any anomalies or deviations from the expected ideal conditions.",
            llm="databricks/databricks-meta-llama-3-3-70b-instruct" )

        llama_runner = LlamaCrewAgent()
        def llama_task_runner(input):
            return llama_runner(input)

        task = Task(
            agent=llama_agent,
            input="You are being given serialized telemetry data from a pharmaceutical bioreactor run. Provide a concise, mechanistic explanation of the run data, any concerning metrics, and your assessment of the cause for any anomalies.",
            run=llama_task_runner,
            description="Bioreactor Troubleshooting.",
            expected_output="A 3-4 sentence assessment of the bioreactor telemetry data and statistically detected anomalies, including root cause analysis. Telemetry: {telemetry_serialized} Anomalies: {anomalies_serialized}. Specifically recommend actions and a high level categorization of the root cause.")
        
        crew = Crew(
            agents=[llama_agent],
            tasks=[task],
            verbose=False)

        result = crew.kickoff()
        print(result)
        return result
    
# ---------------------------
# True Agent wrapper (Databricks Llama)
# ---------------------------
class LlamaRootCauseAgent:
    def __init__(self, endpoint="databricks-meta-llama-3-3-70b-instruct"):
        self.client = mlflow.deployments.get_deploy_client("databricks")
        self.endpoint = endpoint

    def analyze(self, observation) -> dict:
        """Ask the LLM to analyze telemetry and anomalies."""
        messages = [
            {"role": "system", "content": "You are a bioprocess expert. Explain anomalies in bioreactor telemetry and recommend operator actions."},
            {"role": "user", "content": f"""
            Telemetry: {observation.telemetry}
            Anomalies: {[str(a.__dict__) for a in observation.recent_anomalies]}
            Time: {observation.time} h
            Assay budget remaining: {observation.available_budget['assays']}

            Respond ONLY in JSON with fields:
            root_cause: str
            recommended_action: str
            priority: int (1=low, 5=critical)
            rationale: str
            """}
        ]

        response = self.client.predict(
            endpoint=self.endpoint,
            inputs={"messages": messages, "temperature": 0.2, "max_tokens": 300}
        )

        try:
            return json.loads(response["choices"][0]["message"]["content"])
        except Exception:
            return {
                "root_cause": "Unknown",
                "recommended_action": "No action",
                "priority": 1,
                "rationale": "Model output not parsed as JSON."
            }