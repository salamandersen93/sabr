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
        # Inject Streamlit secrets into environment variables for MLflow
        import streamlit as st
        os.environ["DATABRICKS_HOST"] = st.secrets["DATABRICKS_HOST"]
        os.environ["DATABRICKS_TOKEN"] = st.secrets["DATABRICKS_TOKEN"]

        # Create MLflow deploy client (authenticated)
        self.client = mlflow.deployments.get_deploy_client("databricks")
        self.endpoint = endpoint

    def _serialize_df(self, df):
        """Convert DataFrame to compact JSON string."""
        if isinstance(df, pd.DataFrame):
            return df.to_json(orient="records")
        return str(df)

    def explain(self, telemetry_snapshot, anomalies):
        # Serialize telemetry
        if isinstance(telemetry_snapshot, dict):
            telemetry_serialized = {k: self._serialize_df(v) for k, v in telemetry_snapshot.items()}
        elif isinstance(telemetry_snapshot, list):
            telemetry_serialized = [self._serialize_df(v) for v in telemetry_snapshot]
        else:
            raise TypeError("telemetry_snapshot must be a dict or list")
        
        anomalies_serialized = [self._serialize_df(a) for a in anomalies]

        # Define agent
        llama_agent = Agent(
            role="Pharmaceutical Large Molecule Bioreactor Troubleshooting Expert",
            goal="Give a concise, mechanistic explanation of why given conditions and issues might arise in a fed-batch CHO culture.",
            backstory="You are a bioprocess expert. Analyze the following CHO cellbioreactor conditions and provide possible explanations. Categorize the primary root causes of any anomalies or deviations from the expected ideal conditions.",
            llm=self.endpoint
        )

        llama_runner = LlamaCrewAgent()
        def llama_task_runner(input):
            return llama_runner(input)

        task = Task(
            agent=llama_agent,
            input=f"You are being given serialized telemetry data from a pharmaceutical bioreactor run. Provide a concise, mechanistic explanation of the run data, any concerning metrics, and your assessment of the cause for any anomalies.\nTelemetry: {telemetry_serialized}\nAnomalies: {anomalies_serialized}",
            run=llama_task_runner,
            description="Bioreactor Troubleshooting.",
            expected_output="A 3-4 sentence assessment of the bioreactor telemetry data and statistically detected anomalies, including root cause analysis. Specifically recommend actions and a high level categorization of the root cause."
        )

        crew = Crew(
            agents=[llama_agent],
            tasks=[task],
            verbose=False
        )

        result = crew.kickoff()
        print(result)
        return result