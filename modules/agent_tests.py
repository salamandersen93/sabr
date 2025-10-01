from crewai import Agent, Task, Crew
import mlflow.deployments

class DatabricksLLM:
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
        # CrewAI expects a text string back
        return response[0]['candidates'][0]['text'] if isinstance(response, list) else str(response)

# Inject your Databricks model into the agent
databricks_llm = DatabricksLLM()

llama_agent = Agent(
    role="Llama Assistant",
    goal="Answer user queries using the Llama model",
    backstory="A helpful AI assistant powered by Databricks Llama endpoint.",
    llm="databricks/databricks-meta-llama-3-3-70b-instruct" 
)

task = Task(
    agent=llama_agent,
    input="this is a test",
    description="Test the LlamaCrewAgent with a simple prompt.",
    expected_output="A string response from the Llama model."
)

crew = Crew(
    agents=[llama_agent],
    tasks=[task]
)

result = crew.kickoff()
print(result)
