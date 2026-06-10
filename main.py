from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel,Field
import os   


load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from tavily import TavilyClient

tavily_client = TavilyClient()

class Source ( BaseModel):
    """
    Schema for a source used by the agent
    """
    url: str = Field(description="The URL of the source")

class AgentResponse ( BaseModel):
    """
    Schema for the agent's response
    """
    answer: str = Field( description="The answer provided by the agent")
    sources: List[Source] = Field(default_factory=list, description="List of sources used by the agent")

@tool
def search(query: str) -> str:
    """
    Tool that search over internet
    Args:        query (str): The search query
    Returns:        str: The search results
    """
    print(f"Searching for: {query}")
    return tavily_client.search(query)


llm = ChatOpenAI()
tools = [search]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Welcome to the LangSmith Agent Example!")
    result = agent.invoke({"messages": [HumanMessage(content="Give me 2 motivation quotes based on the volleyball sport")]})
    print(result)
    
    
if __name__ == "__main__":
    main()