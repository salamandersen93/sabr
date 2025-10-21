"""
agent_copilot.py

Multi-agent copilot system for bioreactor management.
Observes telemetry, flags anomalies, suggests interventions, and generates reports.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
import mlflow.deployments
import json
import mlflow
from crewai import Agent, Task, Crew
from litellm import completion
from databricks.sdk import WorkspaceClient
import streamlit as st
import os
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun


class DatabricksLiteLLM(LLM):
    """Custom LangChain LLM wrapper for Databricks using LiteLLM."""
    
    token: str = ""
    api_base: str = ""
    model: str = "databricks/databricks-meta-llama-3-3-70b-instruct"
    temperature: float = 0.1
    max_tokens: int = 512
    
    class Config:
        arbitrary_types_allowed = True
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> str:
        """Call the Databricks endpoint using LiteLLM."""
        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                databricks_api_key=self.token,
                databricks_api_base=self.api_base,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract the response content
            if hasattr(response, 'choices') and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                return str(response)
                
        except Exception as e:
            print(f"Error in LiteLLM completion: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM type."""
        return "databricks-litellm"
    
    @property
    def _identifying_params(self) -> Dict:
        """Return identifying parameters."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_base": self.api_base
        }

        
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
        os.environ["DATABRICKS_API_KEY"] = token
        os.environ["DATABRICKS_API_BASE"] = f"{host}/api/2.0"
        
        self.client = WorkspaceClient(host=host, token=token)
        self.token = token
        self.api_base = f"{host}/api/2.0"
        self.model_name = f"databricks/{endpoint}"
        
        # DEBUGGING
        try:
            endpoints = self.client.serving_endpoints.list()
            print("Serving endpoints available:", [e.name for e in endpoints])
            endpoint_info = self.client.serving_endpoints.get(name=endpoint)
            print(f"Endpoint '{endpoint}' status:", endpoint_info.state.ready)
        except Exception as e:
            print(f"Warning: Could not fetch endpoint info: {e}")
        
        print("DATABRICKS_HOST:", os.environ.get("DATABRICKS_HOST"))
        print("DATABRICKS_API_BASE:", os.environ.get("DATABRICKS_API_BASE"))
        
        try:
            # Create the custom LLM instance using LiteLLM
            self.llm = DatabricksLiteLLM(
                token=self.token,
                api_base=self.api_base,
                model=self.model_name,
                temperature=0.1,
                max_tokens=512
            )
            print(f"✓ Successfully initialized LLM with model: {self.model_name}")
            
            # Test the LLM with a simple call
            test_response = self.llm._call("Hello, respond with 'OK' if you can see this.")
            print(f"✓ LLM test call successful: {test_response[:50]}...")
            
        except Exception as e:
            print(f"✗ Error initializing LLM: {e}")
            import traceback
            traceback.print_exc()
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
            print(f"✗ {error_msg}")
            return error_msg
        
        # Convert telemetry list to DataFrame then serialize
        print('Processing telemetry snapshots:', len(telemetry_snapshot), 'records')
        print('Processing anomalies:', len(anomalies), 'records')
        
        if len(telemetry_snapshot) > 0:
            print('Telemetry sample:', telemetry_snapshot[0])
        if len(anomalies) > 0:
            print('Anomalies sample:', anomalies[0])

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
        
        print(f"Serialized {len(telemetry_df)} telemetry records and {len(anomalies_df)} anomalies")

        # Define agent
        try:
            llama_agent = Agent(
                role="Pharmaceutical Large Molecule Bioreactor Troubleshooting Expert",
                goal="Give a concise, mechanistic explanation of why given conditions and issues might arise in a fed-batch CHO culture.",
                backstory="You are a bioprocess expert. Analyze the following CHO cell bioreactor conditions and provide possible explanations. Categorize the primary root causes of any anomalies or deviations from the expected ideal conditions.",
                llm=self.llm,
                verbose=False
            )
            print("✓ Agent created successfully")
        except Exception as e:
            error_msg = f"Error creating agent: {e}"
            print(f"✗ {error_msg}")
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
            print("✓ Agent explanation completed")
            print(result)
            return result
        except Exception as e:
            error_msg = f"Error during crew execution: {e}"
            print(f"✗ {error_msg}")
            import traceback
            traceback.print_exc()
            return error_msg