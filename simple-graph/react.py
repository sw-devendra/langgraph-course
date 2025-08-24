from dotenv import load_dotenv

load_dotenv()
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


@tool
def tripple(x: float) -> float:
    """Return the triple of a number."""
    return 3 * x


tools = [tripple, TavilySearch(max_results=1)]

llm = ChatOpenAI(model="gpt-5-nano", temperature=0).bind_tools(tools)
print("LLM")
