import requests
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from crewai import Agent, Task
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

llama_agent = Agent(
    role="Llama Assistant",
    goal="Answer user queries using the Llama model",
    backstory="A helpful AI assistant powered by Databricks Llama endpoint."
)

llama_runner = LlamaCrewAgent()

def llama_task_runner(input):
    return llama_runner(input)

task = Task(
    agent=llama_agent,
    input="this is a test",
    run=llama_task_runner,
    description="Test the LlamaCrewAgent with a simple prompt.",
    expected_output="A string response from the Llama model."
)
result = task.execute()
display(result)