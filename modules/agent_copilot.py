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
from langchain_community.llms import LiteLLM
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
        print('initializing agentic analysis')
        os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
        os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
        os.environ["OTEL_SDK_DISABLED"] = "true"
        os.environ["DATABRICKS_HOST"] = host
        os.environ["DATABRICKS_TOKEN"] = token
        os.environ["MLFLOW_TRACKING_URI"] = "databricks"
        
        # Critical: Set these for LiteLLM to work with Databricks
        os.environ["DATABRICKS_API_KEY"] = token
        os.environ["DATABRICKS_API_BASE"] = f"{host}/serving-endpoints"
        
        self.client = WorkspaceClient(host=host, token=token)
        # Remove the "databricks/" prefix for the endpoint parameter
        # We'll add it back when creating the LLM
        self.endpoint_name = endpoint
        
        try:
            # Create the CrewAI LLM instance
            # Format: databricks/<endpoint-name>
            self.llm = LiteLLM(model=self.endpoint_name,host=host,token=token,temperature=0.1,max_tokens=512)
            print(f"Successfully initialized CrewAI LLM with endpoint: databricks/{self.endpoint_name}")
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            print(f"  Host: {host}")
            print(f"  Endpoint: databricks/{self.endpoint_name}")
            # Set to None so we can check later
            self.llm = None

    def _serialize_df(self, df):
        """Convert DataFrame to compact JSON string."""
        if isinstance(df, pd.DataFrame):
            return df.to_json(orient="split", index=False)
        return str(df)

    def explain(self, telemetry_snapshot, anomalies):
        # Check if LLM was initialized successfully
        if self.llm is None:
            error_msg = "LLM was not initialized properly. Check Databricks credentials and endpoint configuration."
            print(f" {error_msg}")
            return error_msg
        
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

        # Define agent
        try:
            llama_agent = Agent(
                role="Pharmaceutical Large Molecule Bioreactor Troubleshooting Expert",
                goal="Give a concise, mechanistic explanation of why given conditions and issues might arise in a fed-batch CHO culture.",
                backstory="You are a bioprocess expert. Analyze the following CHO cell bioreactor conditions and provide possible explanations. Categorize the primary root causes of any anomalies or deviations from the expected ideal conditions.",
                llm=self.llm,
                verbose=False
            )
            print(" Agent created successfully")
        except Exception as e:
            error_msg = f"Error creating agent: {e}"
            print(f" {error_msg}")
            import traceback
            traceback.print_exc()
            return error_msg

        task = Task(
            agent=llama_agent,
            description="Bioreactor Troubleshooting. Analyze the serialized telemetry data from a pharmaceutical bioreactor run. Provide a concise, mechanistic explanation of the run data, any concerning metrics, and your assessment of the cause for any anomalies.\nTelemetry: {telemetry}\nAnomalies: {anomalies}",
            expected_output="A 3-4 sentence assessment of the bioreactor telemetry data and statistically detected anomalies, including root cause analysis. Specifically recommend actions and a high level categorization of the root cause."
        )

        crew = Crew(agents=[llama_agent], tasks=[task], verbose=False)
        
        # Pass data as inputs to the crew
        try:
            print("Starting crew execution...")
            result = crew.kickoff(inputs={
                "telemetry": telemetry_serialized,
                "anomalies": anomalies_serialized
            })
            print(" Agent explanation completed")
            print(result)
            return result
        except Exception as e:
            error_msg = f"Error during crew execution: {e}"
            print(f" {error_msg}")
            import traceback
            traceback.print_exc()
            return error_msg