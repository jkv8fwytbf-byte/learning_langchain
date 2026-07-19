from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()

# tool = search engine you can .invoke(...)
tool = TavilySearch(max_results=3)

# in: {"query": "..."}  →  out: search hits (dict with results)
results = tool.invoke({"query": "current temperature in New York"})

print(results)
