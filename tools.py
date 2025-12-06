"""
This file contains the code for the tools the agent will be using
1. retriever_tool
2. web_search_tool
"""

from langchain_core.tools import tool, BaseTool
from langchain_core.vectorstores import VectorStoreRetriever  # imported for type validation in tool (security)
from langchain_community.tools import DuckDuckGoSearchRun
from pydantic import Field

from retriever import CustomRetriever  # custom import


# 1. Retriever Tool: tool to retrieve relevant information from user's notes
# New approach
class CustomRetrieverTool(BaseTool):
    name: str = "retriever_tool"
    description : str = """
    Tool to retrieve the most relevant information about the user's notes
    Args:
        query (str): The user's query to search in the notes
        retriever (VectorStoreRetriever): The VectorStoreRetriever instance to use for retrieval
    Returns:
        str: The retrieved information from the notes most relevant to the query OR no relevant info found
    """
    retriever : VectorStoreRetriever = Field(exclude=True)
    def _run(self, query: str) -> str:
        results = []
        try:
            response = self.retriever.invoke(query)
            if not response:
                return "No relevant information found in the notes."
            for i, result in enumerate(response):
                results.append(f"Document {i + 1}:\n{result.page_content}\n")
            return "\n\n".join(results)
        except Exception as e:
            return f"[retrieval_tool Error]: {str(e)}"




# 2. Web Search Tool: tool to perform web searches
@tool
def web_search_tool(query: str) -> str:
    """This tool is helpful to search the internet or web for more information"""
    search_tool = DuckDuckGoSearchRun()
    results = search_tool.invoke(query)
    print("[web_search_tool] Search Results:")
    return str(results)