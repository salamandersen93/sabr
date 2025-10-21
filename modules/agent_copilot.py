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
from crewai import Agent, Task, Crew, LLM
import mlflow.deployments
from databricks.sdk import WorkspaceClient
import streamlit as st
import os
        
# AGENTIC PROCESSES
class ExplainerAgent:
    def __init__(self, host: str, token: str,
                 endpoint="databricks-meta-llama-3-3-70b-instruct"):
        os.environ["DATABRICKS_HOST"] = host
        os.environ["DATABRICKS_TOKEN"] = token
        
        self.endpoint_name = endpoint
        self.client = mlflow.deployments.get_deploy_client("databricks")
    
    def _call_llm(self, messages):
        response = self.client.predict(
            endpoint=self.endpoint_name,
            inputs={
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 512})
        
        return response['choices'][0]['message']['content']

    def _serialize_df(self, df):
        """Convert DataFrame to compact JSON string."""
        if isinstance(df, pd.DataFrame):
            return df.to_json(orient="split", index=False)
        return str(df)

    def explain(self, telemetry_snapshot, anomalies):
        # Check if LLM was initialized successfully
        
        # Convert telemetry list to DataFrame then serialize
        print('telemetry snapshots:', type(telemetry_snapshot))
        print('anomalies:', type(anomalies))
        
        if len(telemetry_snapshot) > 0:
            print('telemetry sample:', telemetry_snapshot[0])
        if len(anomalies) > 0:
            print('anomalies sample:', anomalies[0])

        telemetry_df = pd.DataFrame(telemetry_snapshot)
        telemetry_serialized = self._serialize_df(telemetry_df)

        anomaly_data = [
            {
                'time': a.time,
                'signal': a.signal,
                'score': a.score,
                'method': a.method,
            }
            for a in anomalies if a.is_anomaly
        ]
        
        anomalies_df = pd.DataFrame(anomaly_data) if anomaly_data else pd.DataFrame()
        anomalies_serialized = self._serialize_df(anomalies_df)

        prompt = f"""
        
            role: Pharmaceutical Large Molecule Bioreactor Troubleshooting Expert.

            goal: Give a concise, mechanistic explanation of why given conditions and issues might arise in a fed-batch CHO culture.

            backstory: You are a bioprocess expert. Analyze the following CHO cell bioreactor conditions and provide possible explanations. Categorize the primary root causes of any anomalies or deviations from the expected ideal conditions.

            Analyze the serialized telemetry data from a pharmaceutical bioreactor run. Provide a concise, mechanistic explanation of the run data, any concerning metrics, and your assessment of the cause for any anomalies.

            Telemetry Data:
            {telemetry_serialized}
            Detected Anomalies:
            {anomalies_serialized}

            Provide a 3-4 sentence assessment including root cause analysis, specific recommended actions, and a high-level categorization of the root cause."""

        messages = [{"role": "user", "content": prompt}]

        try:
            print("Starting LLM call...")
            result = self._call_llm(messages)
            print("Agent explanation completed")
            print(result)
            return result
        
        except Exception as e:
            error_msg = f"Error during LLM call: {e}"
            print(error_msg)
            return error_msg