from dotenv import load_dotenv
import os   


load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from tavily import TavilyClient

tavily_client = TavilyClient()

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
agent = create_agent(model=llm, tools=tools)

def main():
    print("Welcome to the LangSmith Agent Example!")
    result = agent.invoke({"messages": [HumanMessage(content="Give me 2 motivation quotes based on the volleyball sport")]})
    print(result)
    
if __name__ == "__main__":
    main()