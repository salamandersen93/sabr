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
from databricks.sdk import WorkspaceClient
import streamlit as st
import os

class LlamaCrewAgent:
    def __init__(self, endpoint="databricks/databricks-meta-llama-3-3-70b-instruct"):
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
    def __init__(self, host: str, token:str,
                 endpoint="databricks/databricks-meta-llama-3-3-70b-instruct"):
        
        # Set environment variables for LiteLLM/CrewAI to use
        # This is critical for CrewAI to authenticate with Databricks
        os.environ["DATABRICKS_HOST"] = host
        os.environ["DATABRICKS_TOKEN"] = token
        
        # Also set these for MLflow
        os.environ["MLFLOW_TRACKING_URI"] = "databricks"
        
        self.client = WorkspaceClient(host=host, token=token)
        self.endpoint = endpoint

    def _serialize_df(self, df):
        """Convert DataFrame to compact JSON string."""
        if isinstance(df, pd.DataFrame):
            return df.to_json(orient="split", index=False)
        return str(df)

    def explain(self, telemetry_snapshot, anomalies):
        # Convert telemetry list to DataFrame then serialize
        print('telemetry snapshots:', type(telemetry_snapshot))
        print('anomalies:', type(anomalies))
        print('telemetry:', telemetry_snapshot[0])
        print('anomalies:', anomalies[0])
        print('telemetry 0 type:', type(telemetry_snapshot[0]))
        print('anomalies 0 type:', type(anomalies[0]))

        telemetry_df = pd.DataFrame(telemetry_snapshot)
        telemetry_serialized = self._serialize_df(telemetry_df)

        anomaly_data = [{'time': a.time,'signal': a.signal,'score': a.score,'method': a.method,}
                        for a in anomalies if a.is_anomaly]
        
        anomalies_serialized = self._serialize_df(anomalies_df)

        # Define agent - use the endpoint string, LiteLLM will handle it with env vars
        llama_agent = Agent(
            role="Pharmaceutical Large Molecule Bioreactor Troubleshooting Expert",
            goal="Give a concise, mechanistic explanation of why given conditions and issues might arise in a fed-batch CHO culture.",
            backstory="You are a bioprocess expert. Analyze the following CHO cellbioreactor conditions and provide possible explanations. Categorize the primary root causes of any anomalies or deviations from the expected ideal conditions.",
            llm=self.endpoint,
            verbose=False # Optional: helps with debugging
        )

        task = Task(
            agent=llama_agent,
            description="Bioreactor Troubleshooting. Analyze the serialized telemetry data from a pharmaceutical bioreactor run. Provide a concise, mechanistic explanation of the run data, any concerning metrics, and your assessment of the cause for any anomalies.\nTelemetry: {telemetry}\nAnomalies: {anomalies}",
            expected_output="A 3-4 sentence assessment of the bioreactor telemetry data and statistically detected anomalies, including root cause analysis. Specifically recommend actions and a high level categorization of the root cause."
        )
        task = Task(
            agent=llama_agent,
            description="Bioreactor Troubleshooting. Analyze the serialized telemetry data from a pharmaceutical bioreactor run. Provide a concise, mechanistic explanation of the run data, any concerning metrics, and your assessment of the cause for any anomalies.\nTelemetry: {telemetry}\nAnomalies: {anomalies}",
            expected_output="A 3-4 sentence assessment of the bioreactor telemetry data and statistically detected anomalies, including root cause analysis. Specifically recommend actions and a high level categorization of the root cause."
        )

        crew = Crew(agents=[llama_agent],tasks=[task],verbose=False)
        # Pass data as inputs to the crew
        result = crew.kickoff(inputs={
            "telemetry": telemetry_serialized,
            "anomalies": anomalies_serialized})
        
        print(result)
        return result