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


class LlamaCrewAgent:
    def __init__(self, endpoint="databricks/databricks-meta-llama-3-1-8b-instruct"):
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
    def __init__(self, host: str, token: str,
                 endpoint="databricks/databricks-meta-llama-3-3-70b-instruct"):
        print('initializing agentic analysis')
        os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
        os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
        os.environ["OTEL_SDK_DISABLED"] = "true"
        os.environ["DATABRICKS_HOST"] = host
        os.environ["DATABRICKS_TOKEN"] = token
        os.environ["MLFLOW_TRACKING_URI"] = "databricks"
        os.environ["DATABRICKS_API_KEY"] = token
        
        # For LiteLLM/CrewAI, we also need DATABRICKS_API_BASE
        # This should be the serving endpoints URL
        os.environ["DATABRICKS_API_BASE"] = f"{host}/serving-endpoints"
        
        self.client = WorkspaceClient(host=host, token=token)
        self.endpoint = endpoint
        
        # Create the CrewAI LLM instance with proper Databricks configuration
        # The model name should be in the format: databricks/<model-name>
        self.llm = LLM(
            model=endpoint,
            temperature=0.1,
            max_tokens=512
        )
        print(f"Initialized CrewAI LLM with endpoint: {endpoint}")

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

        # Define agent - use the CrewAI LLM instance
        llama_agent = Agent(
            role="Pharmaceutical Large Molecule Bioreactor Troubleshooting Expert",
            goal="Give a concise, mechanistic explanation of why given conditions and issues might arise in a fed-batch CHO culture.",
            backstory="You are a bioprocess expert. Analyze the following CHO cell bioreactor conditions and provide possible explanations. Categorize the primary root causes of any anomalies or deviations from the expected ideal conditions.",
            llm=self.llm,
            verbose=False
        )

        task = Task(
            agent=llama_agent,
            description="Bioreactor Troubleshooting. Analyze the serialized telemetry data from a pharmaceutical bioreactor run. Provide a concise, mechanistic explanation of the run data, any concerning metrics, and your assessment of the cause for any anomalies.\nTelemetry: {telemetry}\nAnomalies: {anomalies}",
            expected_output="A 3-4 sentence assessment of the bioreactor telemetry data and statistically detected anomalies, including root cause analysis. Specifically recommend actions and a high level categorization of the root cause."
        )

        crew = Crew(agents=[llama_agent], tasks=[task], verbose=False)
        
        # Pass data as inputs to the crew
        try:
            result = crew.kickoff(inputs={
                "telemetry": telemetry_serialized,
                "anomalies": anomalies_serialized
            })
            print("Agent explanation result:")
            print(result)
            return result
        except Exception as e:
            print(f"Error during crew execution: {e}")
            import traceback
            traceback.print_exc()
            return f"Error generating explanation: {str(e)}"